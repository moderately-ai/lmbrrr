//! Unit tests for the decode-state machinery every speculative round relies
//! on: the truncatable KV cache, tree-mask visibility rules, and the
//! closed-form verify-state reconstruction. All CPU-only constructions.

use super::*;

fn kv(values: &[f32], rows: usize) -> Tensor {
    // (b=1, h=1, rows, d=2)
    Tensor::from_slice(values, (1, 1, rows, 2), &Device::Cpu).unwrap()
}

fn kv_rows(cache_side: &Tensor, len: usize) -> Vec<f32> {
    cache_side
        .narrow(2, 0, len)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

#[test]
fn kv_cache_append_truncate_overwrite() {
    let mut cache = TruncatableKvCache::new();
    assert_eq!(cache.len(), 0);

    let (k, _v) = cache
        .append(&kv(&[1., 2., 3., 4.], 2), &kv(&[5., 6., 7., 8.], 2))
        .unwrap();
    assert_eq!(cache.len(), 2);
    assert_eq!(kv_rows(&k, 2), [1., 2., 3., 4.]);

    // Rewind one row; the next append must overwrite the stale slice.
    cache.truncate(1).unwrap();
    assert_eq!(cache.len(), 1);
    let (k, v) = cache
        .append(&kv(&[9., 10.], 1), &kv(&[11., 12.], 1))
        .unwrap();
    assert_eq!(cache.len(), 2);
    assert_eq!(kv_rows(&k, 2), [1., 2., 9., 10.]);
    assert_eq!(kv_rows(&v, 2), [5., 6., 11., 12.]);

    assert!(cache.truncate(3).is_err(), "cannot truncate forward");
}

#[test]
fn kv_cache_grows_past_min_capacity_preserving_rows() {
    let mut cache = TruncatableKvCache::new();
    // Push past MIN_CAPACITY (1024) so ensure_capacity reallocates and must
    // copy the retained prefix.
    let chunk: Vec<f32> = (0..2 * 600).map(|i| i as f32).collect();
    cache
        .append(&kv(&chunk, 600), &kv(&chunk, 600))
        .unwrap();
    let (k, _) = cache.append(&kv(&chunk, 600), &kv(&chunk, 600)).unwrap();
    assert_eq!(cache.len(), 1200);
    let rows = kv_rows(&k, 1200);
    // Prefix preserved across the reallocation; the second chunk repeats the
    // same values, so the buffer tail is the chunk tail again.
    assert_eq!(&rows[..4], &[0., 1., 2., 3.]);
    assert_eq!(&rows[1200 * 2 - 2..], &[1198., 1199.]);
}

#[test]
fn kv_cache_compact_rows_moves_alt_branch_down() {
    let mut cache = TruncatableKvCache::new();
    // Layout: history(2) + anchor(1) + a-branch(2) + b-branch(2) = 7 rows.
    let rows: Vec<f32> = (0..7 * 2).map(|i| i as f32).collect();
    cache.append(&kv(&rows, 7), &kv(&rows, 7)).unwrap();

    // Alternate wins with 2 accepted: rows [5, 6] move down to [3, 4],
    // length becomes 5 (history + anchor + 2 winner rows).
    cache.compact_rows(3, 5, 2).unwrap();
    assert_eq!(cache.len(), 5);
    let (k, _) = cache.append(&kv(&[99., 99.], 1), &kv(&[99., 99.], 1)).unwrap();
    let got = kv_rows(&k, 6);
    assert_eq!(&got[..6], &[0., 1., 2., 3., 4., 5.], "history + anchor untouched");
    assert_eq!(&got[6..10], &[10., 11., 12., 13.], "b rows compacted down");
    assert_eq!(&got[10..12], &[99., 99.]);

    assert!(cache.compact_rows(5, 3, 1).is_err(), "dst must be < src");
    assert!(cache.compact_rows(0, 5, 3).is_err(), "range beyond len");
}

#[test]
fn kv_cache_compact_rows_multi_head() {
    // heads > 1 makes the dim-2 narrow strided — the production shape;
    // regression for slice_set rejecting the non-contiguous source (copy()
    // clones storage but keeps the narrow's layout).
    let mut cache = TruncatableKvCache::new();
    let rows: Vec<f32> = (0..2 * 7 * 2).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&rows, (1, 2, 7, 2), &Device::Cpu).unwrap();
    cache.append(&t, &t).unwrap();

    cache.compact_rows(3, 5, 2).unwrap();
    assert_eq!(cache.len(), 5);
    let probe = Tensor::from_slice(&[99f32, 99., 98., 98.], (1, 2, 1, 2), &Device::Cpu).unwrap();
    let (k, _) = cache.append(&probe, &probe).unwrap();
    let got = kv_rows(&k, 6);
    assert_eq!(&got[..6], &[0., 1., 2., 3., 4., 5.], "head-0 prefix untouched");
    assert_eq!(&got[6..10], &[10., 11., 12., 13.], "head-0 b rows compacted");
    assert_eq!(&got[10..12], &[99., 99.]);
    assert_eq!(&got[12..18], &[14., 15., 16., 17., 18., 19.], "head-1 prefix untouched");
    assert_eq!(&got[18..22], &[24., 25., 26., 27.], "head-1 b rows compacted");
    assert_eq!(&got[22..24], &[98., 98.]);
}

#[test]
fn tree_mask_visibility_rules() {
    // w = 2, offset = 3: rows [anchor, a1, a2, b1, b2], cols
    // [h0 h1 h2 | anchor a1 a2 b1 b2].
    let w = 2;
    let offset = 3;
    let l = 1 + 2 * w;
    let total = offset + l;
    let data = tree_mask_data(w, offset);
    assert_eq!(data.len(), l * total);
    let visible = |row: usize, col: usize| data[row * total + col] == 0.0;

    for row in 0..l {
        for col in 0..=offset {
            assert!(visible(row, col), "row {row} must see history+anchor col {col}");
        }
    }
    // a2 (row 2) sees a1 and itself, not b-rows.
    assert!(visible(2, offset + 1) && visible(2, offset + 2));
    assert!(!visible(2, offset + 3) && !visible(2, offset + 4), "a2 must not see b rows");
    // b2 (row 4) sees b1 and itself, not a-rows.
    assert!(visible(4, offset + 3) && visible(4, offset + 4));
    assert!(!visible(4, offset + 1) && !visible(4, offset + 2), "b2 must not see a rows");
    // anchor row sees nothing beyond itself.
    for col in offset + 1..total {
        assert!(!visible(0, col));
    }
}

/// Closed-form capture reconstruction must equal the sequential recurrence
/// S_i = exp(g_i) * S_{i-1} + k_i (x) delta_i at every prefix, in both the
/// v1 (dk, dv) and v2 transposed (dv, dk) layouts, and the conv window must
/// slice/pad exactly like the running conv state would.
#[test]
fn reconstruct_capture_state_matches_sequential_recurrence() {
    let device = Device::Cpu;
    let (h, dk, dv, c, ksz) = (2usize, 3usize, 3usize, 3usize, 4usize);

    let val = |i: usize| ((i * 37 % 19) as f32 - 9.0) / 7.0;
    let s0_v: Vec<f32> = (0..h * dk * dv).map(val).collect();
    let kc_v: Vec<f32> = (0..h * c * dk).map(|i| val(i + 100)).collect();
    let de_v: Vec<f32> = (0..h * c * dv).map(|i| val(i + 200)).collect();
    // Log-decays g_i in (-1, 0); gcs is the inclusive cumsum.
    let g: Vec<f32> = (0..h * c).map(|i| -0.1 - 0.07 * ((i % 5) as f32)).collect();
    let mut gcs_v = vec![0.0f32; h * c];
    for head in 0..h {
        let mut acc = 0.0;
        for i in 0..c {
            acc += g[head * c + i];
            gcs_v[head * c + i] = acc;
        }
    }

    let s0 = Tensor::from_slice(&s0_v, (1, h, dk, dv), &device).unwrap();
    let kc = Tensor::from_slice(&kc_v, (1, h, c, dk), &device).unwrap();
    let delta = Tensor::from_slice(&de_v, (1, h, c, dv), &device).unwrap();
    let gcs = Tensor::from_slice(&gcs_v, (1, h, c), &device).unwrap();

    // conv_full: prev state (ksz cols) + c new cols over conv_dim=2 channels.
    let conv_dim = 2usize;
    let conv_v: Vec<f32> = (0..conv_dim * (ksz + c)).map(|i| val(i + 300)).collect();
    let conv_full = Tensor::from_slice(&conv_v, (1, conv_dim, ksz + c), &device).unwrap();

    // Sequential reference per head.
    let mut state = s0_v.clone();
    let mut references: Vec<Vec<f32>> = Vec::new();
    for i in 0..c {
        let mut next = vec![0.0f32; h * dk * dv];
        for head in 0..h {
            let decay = g[head * c + i].exp();
            for a in 0..dk {
                for b in 0..dv {
                    let idx = head * dk * dv + a * dv + b;
                    next[idx] = decay * state[idx]
                        + kc_v[head * c * dk + i * dk + a] * de_v[head * c * dv + i * dv + b];
                }
            }
        }
        state = next;
        references.push(state.clone());
    }

    // The Lazy variant defers the transpose+cat to the rollback; it must
    // produce byte-identical windows to the assembled Full form.
    let conv_prev = conv_full.narrow(2, 0, ksz).unwrap();
    let mixed_raw = conv_full
        .narrow(2, ksz, c)
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap();

    for prefix in 1..=c {
        for transposed in [false, true] {
            for lazy in [false, true] {
            let conv_cap = if lazy {
                ConvCapture::Lazy {
                    conv_prev: conv_prev.clone(),
                    mixed_raw: mixed_raw.clone(),
                }
            } else {
                ConvCapture::Full {
                    conv_full: conv_full.clone(),
                    prev_conv_len: ksz,
                }
            };
            let cap = DeltaVerifyCapture {
                s0: if transposed {
                    s0.transpose(2, 3).unwrap().contiguous().unwrap()
                } else {
                    s0.clone()
                },
                kc: kc.clone(),
                delta: delta.clone(),
                gcs: gcs.clone(),
                conv: conv_cap,
                dtype: DType::F32,
                transposed,
            };
            let (rec, conv) =
                GatedDeltaNet::reconstruct_capture_state(&cap, prefix, ksz).unwrap();
            let rec = if transposed {
                rec.transpose(2, 3).unwrap().contiguous().unwrap()
            } else {
                rec
            };
            let got = rec.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let want = &references[prefix - 1];
            for (index, (g_val, w_val)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g_val - w_val).abs() < 1e-4,
                    "prefix {prefix} transposed {transposed} idx {index}: {g_val} vs {w_val}"
                );
            }

            // Conv window: last ksz columns of prev(ksz) + prefix inputs.
            let window = conv.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let start = ksz + prefix - ksz; // = prefix
            for ch in 0..conv_dim {
                for col in 0..ksz {
                    let want = conv_v[ch * (ksz + c) + start + col];
                    let got = window[ch * ksz + col];
                    assert_eq!(
                        got, want,
                        "conv ch {ch} col {col} prefix {prefix} lazy {lazy}"
                    );
                }
            }
            }
        }
    }
}

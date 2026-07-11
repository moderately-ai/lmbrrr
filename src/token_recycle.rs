//! Verify-logit token recycling (draft-free speculation, second source).
//!
//! Every verify forward computes full logits at every chunk position and we
//! keep only the argmax. This module banks the top-k candidates (with their
//! logit margins) per input token in an adjacency table, and proposes short
//! chains through the table as drafts when the trained drafter has been
//! judged unprofitable.
//!
//! Design constraints baked in from measurement (see the ticket):
//!
//! * **Margin gating is the viability condition, not a tuning nicety.** On
//!   the measured cost model a depth-1 draft needs ~77% acceptance to beat a
//!   plain greedy step (verify l=2 ≈ 1.77× a decode step). A Markov-1 table
//!   only clears that bar on near-deterministic continuations, which are
//!   exactly the rows with a large top-1/top-2 logit margin. The proposer
//!   therefore extends a chain only while the banked margin stays above the
//!   caller's threshold.
//! * **Short chains.** Chained acceptance compounds; depth is capped by the
//!   caller (default 2 in the runner).
//! * **Most-recent-wins updates.** Rows are overwritten by the latest verify
//!   observation, matching the reference behaviour (locally consistent
//!   continuations beat stale global ones).
//!
//! The table is sparse (only tokens actually seen as verify inputs get
//! rows), so memory is proportional to the generation, not the vocabulary.

use std::collections::HashMap;

/// Candidates banked for one input token, best-first.
#[derive(Clone, Debug)]
pub struct RecycledRow {
    /// Top candidate ids, descending logit order.
    pub candidates: Vec<u32>,
    /// Logit gap between candidate 0 and candidate 1 (f32::INFINITY when
    /// only one candidate was banked). The gating signal: a proxy for how
    /// deterministic the target was at this token last time it was scored.
    pub top_margin: f32,
}

#[derive(Debug, Default)]
pub struct RecycleTable {
    rows: HashMap<u32, RecycledRow>,
    updates: usize,
}

impl RecycleTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Banks the top candidates observed for `input_token`, best-first, with
    /// their logits. Overwrites any previous row (most recent wins).
    pub fn update(&mut self, input_token: u32, ranked: &[(u32, f32)]) {
        if ranked.is_empty() {
            return;
        }
        let top_margin = if ranked.len() >= 2 {
            ranked[0].1 - ranked[1].1
        } else {
            f32::INFINITY
        };
        self.rows.insert(
            input_token,
            RecycledRow {
                candidates: ranked.iter().map(|(id, _)| *id).collect(),
                top_margin,
            },
        );
        self.updates += 1;
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn updates(&self) -> usize {
        self.updates
    }

    pub fn row(&self, token: u32) -> Option<&RecycledRow> {
        self.rows.get(&token)
    }

    /// Proposes a chain of top-1 candidates starting from `current`,
    /// extending only while the banked margin stays at or above
    /// `margin_threshold`, up to `max_depth` tokens. Returns None when even
    /// the first hop fails the gate — the caller falls through to the next
    /// draft source.
    pub fn propose(
        &self,
        current: u32,
        max_depth: usize,
        margin_threshold: f32,
    ) -> Option<Vec<u32>> {
        let mut chain = Vec::new();
        let mut cursor = current;
        while chain.len() < max_depth {
            let Some(row) = self.rows.get(&cursor) else {
                break;
            };
            if row.top_margin < margin_threshold || row.candidates.is_empty() {
                break;
            }
            let next = row.candidates[0];
            // A self-loop above threshold would draft an infinite repeat;
            // permit a single hop of it (repetition happens legitimately)
            // but never chain it.
            if !chain.is_empty() && next == cursor {
                break;
            }
            chain.push(next);
            cursor = next;
        }
        if chain.is_empty() {
            None
        } else {
            Some(chain)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_table_proposes_nothing() {
        let table = RecycleTable::new();
        assert!(table.propose(5, 2, 0.0).is_none());
    }

    #[test]
    fn margin_gate_blocks_uncertain_rows() {
        let mut table = RecycleTable::new();
        table.update(5, &[(7, 10.0), (8, 9.5)]); // margin 0.5
        assert!(table.propose(5, 2, 1.0).is_none());
        assert_eq!(table.propose(5, 2, 0.25).unwrap(), vec![7]);
    }

    #[test]
    fn chain_extends_through_confident_rows_and_stops_at_uncertain() {
        let mut table = RecycleTable::new();
        table.update(5, &[(7, 10.0), (8, 2.0)]); // margin 8 — confident
        table.update(7, &[(9, 11.0), (1, 4.0)]); // margin 7 — confident
        table.update(9, &[(3, 6.0), (4, 5.9)]); // margin 0.1 — uncertain
        let chain = table.propose(5, 4, 5.0).unwrap();
        assert_eq!(chain, vec![7, 9]); // stops before the uncertain hop
    }

    #[test]
    fn depth_cap_respected() {
        let mut table = RecycleTable::new();
        table.update(1, &[(2, 10.0), (0, 0.0)]);
        table.update(2, &[(3, 10.0), (0, 0.0)]);
        table.update(3, &[(4, 10.0), (0, 0.0)]);
        assert_eq!(table.propose(1, 2, 1.0).unwrap(), vec![2, 3]);
    }

    #[test]
    fn most_recent_update_wins() {
        let mut table = RecycleTable::new();
        table.update(5, &[(7, 10.0), (8, 1.0)]);
        table.update(5, &[(9, 12.0), (7, 2.0)]);
        assert_eq!(table.propose(5, 1, 1.0).unwrap(), vec![9]);
    }

    #[test]
    fn self_loop_allowed_once_never_chained() {
        let mut table = RecycleTable::new();
        table.update(5, &[(5, 10.0), (8, 1.0)]); // token predicts itself
        let chain = table.propose(5, 4, 1.0).unwrap();
        assert_eq!(chain, vec![5]); // one hop, no infinite repeat
    }

    #[test]
    fn single_candidate_row_has_infinite_margin() {
        let mut table = RecycleTable::new();
        table.update(5, &[(7, 10.0)]);
        assert_eq!(table.propose(5, 1, 1e9).unwrap(), vec![7]);
    }
}

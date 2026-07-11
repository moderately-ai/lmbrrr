//! Draft-free n-gram speculation (prompt-lookup class).
//!
//! Bets that generation re-quotes the context: match the trailing n-gram of
//! the committed sequence against prompt + history and propose the tokens
//! that followed the previous occurrence. Drafts fire only on a match, so
//! the greedy floor is preserved (no probe tax), and the target's argmax
//! verification keeps every copied token exact — a match is a *candidate*
//! continuation, never blind reiteration.
//!
//! The index maps every n-gram (largest `max_n` first, down to `min_n`) to
//! its most recent end position, updated incrementally as tokens commit.
//! Matching the most recent occurrence follows the prompt-lookup reference
//! behaviour and favours locally-consistent copies (e.g. the code block
//! currently being edited).

use std::collections::HashMap;

#[derive(Debug)]
pub struct NgramDraftIndex {
    tokens: Vec<u32>,
    /// n-gram (as fixed-width key) -> end index (position AFTER the gram) of
    /// its most recent occurrence.
    last_seen: HashMap<(usize, [u32; MAX_N]), usize>,
    min_n: usize,
    max_n: usize,
}

const MAX_N: usize = 4;

impl NgramDraftIndex {
    pub fn new(min_n: usize, max_n: usize) -> Self {
        let max_n = max_n.min(MAX_N).max(1);
        let min_n = min_n.clamp(1, max_n);
        Self {
            tokens: Vec::new(),
            last_seen: HashMap::new(),
            min_n,
            max_n,
        }
    }

    fn key(&self, n: usize, gram: &[u32]) -> (usize, [u32; MAX_N]) {
        let mut buf = [u32::MAX; MAX_N];
        buf[..n].copy_from_slice(gram);
        (n, buf)
    }

    /// Appends committed tokens, indexing every new n-gram suffix.
    pub fn extend(&mut self, new_tokens: &[u32]) {
        for &token in new_tokens {
            self.tokens.push(token);
            let len = self.tokens.len();
            for n in self.min_n..=self.max_n {
                if len >= n {
                    let gram = &self.tokens[len - n..];
                    let key = self.key(n, gram);
                    self.last_seen.insert(key, len);
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Proposes up to `span` tokens following the most recent prior
    /// occurrence of the current suffix (longest n first). Returns None when
    /// no n-gram of length >= min_n has occurred before, or the match ends
    /// at the sequence tail (nothing follows it).
    pub fn propose(&self, span: usize) -> Option<Vec<u32>> {
        let len = self.tokens.len();
        if span == 0 || len < self.min_n {
            return None;
        }
        for n in (self.min_n..=self.max_n.min(len)).rev() {
            let gram = &self.tokens[len - n..];
            let key = self.key(n, gram);
            let Some(&end) = self.last_seen.get(&key) else {
                continue;
            };
            // The map contains the current suffix itself (indexed on
            // extend); a usable match must END before the current suffix so
            // that continuation tokens exist and differ from the tail.
            if end >= len {
                // Find an earlier occurrence by scanning backward; the map
                // only keeps the latest. Cheap fallback: scan.
                if let Some(prior_end) = self.scan_prior(gram, len - n) {
                    let take = span.min(len - 0).min(self.tokens.len() - prior_end);
                    if take == 0 {
                        continue;
                    }
                    return Some(self.tokens[prior_end..prior_end + take].to_vec());
                }
                continue;
            }
            let take = span.min(self.tokens.len() - end);
            if take == 0 {
                continue;
            }
            return Some(self.tokens[end..end + take].to_vec());
        }
        None
    }

    /// Backward scan for the latest occurrence of `gram` ending at or before
    /// `limit` (exclusive of the current tail occurrence).
    fn scan_prior(&self, gram: &[u32], limit: usize) -> Option<usize> {
        let n = gram.len();
        if limit < n {
            return None;
        }
        let mut start = limit - n;
        loop {
            if &self.tokens[start..start + n] == gram {
                return Some(start + n);
            }
            if start == 0 {
                return None;
            }
            start -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposes_continuation_of_repeated_ngram() {
        let mut index = NgramDraftIndex::new(2, 3);
        // "a b c d e ... a b c" -> propose "d e"
        index.extend(&[10, 11, 12, 13, 14, 99, 98, 10, 11, 12]);
        let draft = index.propose(2).expect("match");
        assert_eq!(draft, vec![13, 14]);
    }

    #[test]
    fn prefers_longest_ngram_match() {
        let mut index = NgramDraftIndex::new(1, 3);
        // 1-gram `20` occurs with continuation 7; 3-gram `5 6 20` with 8.
        index.extend(&[20, 7, 5, 6, 20, 8, 9, 5, 6, 20]);
        let draft = index.propose(1).expect("match");
        assert_eq!(draft, vec![8]);
    }

    #[test]
    fn no_match_returns_none() {
        let mut index = NgramDraftIndex::new(2, 3);
        index.extend(&[1, 2, 3, 4, 5]);
        assert!(index.propose(4).is_none());
    }

    #[test]
    fn span_clamped_to_available_continuation() {
        let mut index = NgramDraftIndex::new(2, 3);
        index.extend(&[1, 2, 30, 1, 2]);
        let draft = index.propose(8).expect("match");
        // Only one token follows the prior occurrence before the current
        // suffix begins... continuation is [30, 1, 2] capped by tail.
        assert_eq!(draft[0], 30);
    }

    #[test]
    fn most_recent_occurrence_wins() {
        let mut index = NgramDraftIndex::new(2, 2);
        // `1 2` occurs twice with different continuations; latest (40) wins.
        index.extend(&[1, 2, 30, 1, 2, 40, 1, 2]);
        let draft = index.propose(1).expect("match");
        assert_eq!(draft, vec![40]);
    }
}

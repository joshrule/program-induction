use super::{super::as_result, SampleError, TRS};
use itertools::Itertools;
use rand::Rng;
use term_rewriting::Rule;

impl<'a, 'b> TRS<'a, 'b> {
    /// Selects a rule from the TRS at random, swaps the LHS and RHS if possible and inserts the resulting rules
    /// back into the TRS imediately after the background.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{parse_lexicon, parse_trs};
    /// # use polytype::Context as TypeContext;
    /// # use rand::thread_rng;
    /// let mut rng = thread_rng();
    /// let mut lex = parse_lexicon(
    ///     "PLUS/2: int -> int -> int; SUCC/1: int -> int; ZERO/0: int;",
    ///     TypeContext::default(),
    /// )
    ///     .expect("parsed lexicon");
    ///
    /// let trs1 = parse_trs(
    ///     "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_)) | PLUS(SUCC(x_) y_);",
    ///     &mut lex,
    ///     false,
    ///     &[]
    /// )
    ///     .expect("parsed trs");
    /// assert_eq!(trs1.len(), 1);
    /// assert_eq!(trs1.to_string(), "PLUS(v0_ SUCC(v1_)) = SUCC(PLUS(v0_ v1_)) | PLUS(SUCC(v0_) v1_);");
    ///
    /// let trs2 = trs1.swap_lhs_and_rhs(&mut rng).unwrap();
    /// assert_eq!(trs2.len(), 2);
    /// assert_eq!(trs2.to_string(), "SUCC(PLUS(v0_ v1_)) = PLUS(v0_ SUCC(v1_));\nPLUS(SUCC(v0_) v1_) = PLUS(v0_ SUCC(v1_));");
    /// ```
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{parse_lexicon, parse_trs};
    /// # use polytype::Context as TypeContext;
    /// # use rand::thread_rng;
    /// let mut rng = thread_rng();
    /// let mut lex = parse_lexicon(
    ///     "PLUS/2: int -> int -> int; SUCC/1: int -> int; ZERO/0: int;",
    ///     TypeContext::default(),
    /// )
    ///     .expect("parsed lexicon");
    ///
    /// let trs = parse_trs("PLUS(x_ y_) = SUCC(x_);", &mut lex, true, &[] ).expect("parsed trs");
    /// assert!(trs.swap_lhs_and_rhs(&mut rng).is_err());
    /// ```
    pub fn swap_lhs_and_rhs<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        if !self.is_empty() {
            let idx = rng.gen_range(0, self.len());
            let mut trs = self.clone();
            let mut new_rules = TRS::swap_rule(&trs.utrs.rules[idx])?;
            self.filter_background(&mut new_rules);
            trs.utrs.remove_idx(idx)?;
            trs.utrs.inserts_idx(idx, new_rules)?;
            Ok(trs)
        } else {
            Err(SampleError::OptionsExhausted)
        }
    }
    /// returns a vector of a rules with each rhs being the lhs of the original
    /// rule and each lhs is each rhs of the original.
    fn swap_rule(rule: &Rule) -> Result<Vec<Rule>, SampleError> {
        let rules = rule
            .clauses()
            .iter()
            .filter_map(TRS::swap_clause)
            .collect_vec();
        as_result(rules)
    }
    /// Swap lhs and rhs iff the rule is deterministic and swap is a valid rule.
    fn swap_clause(rule: &Rule) -> Option<Rule> {
        rule.rhs()
            .and_then(|rhs| Rule::new(rhs, vec![rule.lhs.clone()]))
    }
}

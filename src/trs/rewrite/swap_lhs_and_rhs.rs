use super::{super::as_result, SampleError, TRS};
use itertools::Itertools;
use rand::Rng;
use term_rewriting::Rule;

impl<'ctx, 'b> TRS<'ctx, 'b> {
    /// Selects a rule from the TRS at random, swaps the LHS and RHS if possible and inserts the resulting rules
    /// back into the TRS imediately after the background.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # use programinduction::trs::{parse_lexicon, parse_trs};
    /// # use rand::thread_rng;
    /// # use polytype::{Source, atype::{with_ctx, TypeSchema, TypeContext}};
    /// with_ctx(10, |ctx: TypeContext<'_>| {
    ///     let mut rng = thread_rng();
    ///     let mut lex = parse_lexicon(
    ///         "PLUS/2: int -> int -> int; SUCC/1: int -> int; ZERO/0: int;",
    ///         &ctx,
    ///     ).expect("lex");
    ///
    ///     let trs = parse_trs(
    ///         "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_)) | PLUS(SUCC(x_) y_);",
    ///         &mut lex, false, &[],
    ///     ).expect("trs");
    ///     assert_eq!(trs.len(), 1);
    ///
    ///     let trs = trs.swap_lhs_and_rhs(&mut rng).expect("trs");
    ///     assert_eq!(trs.len(), 2);
    ///     assert_eq!(trs.to_string(), "SUCC(PLUS(v0_ v1_)) = PLUS(v0_ SUCC(v1_));\nPLUS(SUCC(v0_) v1_) = PLUS(v0_ SUCC(v1_));");
    ///
    ///     let trs = parse_trs("PLUS(x_ y_) = SUCC(x_);", &mut lex, true, &[] ).expect("trs");
    ///     assert!(trs.swap_lhs_and_rhs(&mut rng).is_err());
    /// })
    /// ```
    pub fn swap_lhs_and_rhs<R: Rng>(&self, rng: &mut R) -> Result<Self, SampleError<'ctx>> {
        if !self.is_empty() {
            let idx = rng.gen_range(0..self.len());
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
    fn swap_rule(rule: &Rule) -> Result<Vec<Rule>, SampleError<'ctx>> {
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

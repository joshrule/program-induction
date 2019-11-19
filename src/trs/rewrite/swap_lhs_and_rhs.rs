use itertools::Itertools;
use rand::Rng;
use term_rewriting::Rule;

use super::{SampleError, TRS};

impl TRS {
    /// Selects a rule from the TRS at random, swaps the LHS and RHS if possible and inserts the resulting rules
    /// back into the TRS imediately after the background.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use polytype::Context as TypeContext;
    /// # use rand::{thread_rng};
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule};
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some(".".to_string()));
    /// ops.push(ptp![0, 1; @arrow[tp!(@arrow[tp!(0), tp!(1)]), tp!(0), tp!(1)]]);
    /// sig.new_op(2, Some("PLUS".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// sig.new_op(1, Some("SUCC".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    /// sig.new_op(0, Some("ZERO".to_string()));
    /// ops.push(ptp![int]);
    ///
    /// let rules = vec![
    ///     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_)) | PLUS(SUCC(x_) y_)").expect("parsed rule"),
    /// ];
    ///
    /// let vars = vec![
    ///     ptp![int],
    ///     ptp![int],
    ///     ptp![int],
    /// ];
    ///
    /// for op in sig.operators() {
    ///     println!("{:?}/{}", op.name(&sig), op.arity())
    /// }
    /// for r in &rules {
    ///     println!("{:?}", r.pretty(&sig));
    /// }
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], vec![], false, TypeContext::default());
    ///
    /// let mut trs = TRS::new(&lexicon, rules).unwrap();
    ///
    /// assert_eq!(trs.len(), 1);
    ///
    /// let mut rng = thread_rng();
    ///
    /// if let Ok(new_trs) = trs.swap_lhs_and_rhs(&mut rng) {
    ///     assert_eq!(new_trs.len(), 2);
    ///     let display_str = format!("{}", new_trs);
    ///     assert_eq!(display_str, "SUCC(PLUS(x_ y_)) = PLUS(x_ SUCC(y_));\nPLUS(SUCC(x_) y_) = PLUS(x_ SUCC(y_));");
    /// } else {
    ///     assert_eq!(trs.len(), 1);
    /// }
    ///
    ///
    /// let mut sig = Signature::default();
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some(".".to_string()));
    /// ops.push(ptp![0, 1; @arrow[tp!(@arrow[tp!(0), tp!(1)]), tp!(0), tp!(1)]]);
    /// sig.new_op(2, Some("A".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// sig.new_op(1, Some("B".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    ///
    /// let rules = vec![
    ///     parse_rule(&mut sig, "A(x_ y_) = B(x_ )").expect("parsed rule"),
    /// ];
    ///
    /// let vars = vec![
    ///     ptp![int],
    ///     ptp![int],
    ///     ptp![int],
    /// ];
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], vec![], false, TypeContext::default());
    ///
    /// let mut trs = TRS::new(&lexicon, rules).unwrap();
    ///
    /// assert!(trs.swap_lhs_and_rhs(&mut rng).is_err());
    /// # }
    /// ```
    pub fn swap_lhs_and_rhs<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let num_rules = self.len();
        let num_background = self
            .lex
            .0
            .read()
            .expect("poisoned lexicon")
            .background
            .len();
        if num_background <= num_rules {
            let idx = rng.gen_range(num_background, num_rules);
            let mut trs = self.clone();
            let new_rules = TRS::swap_rule_helper(&trs.utrs.rules[idx])?;
            trs.utrs.remove_idx(idx)?;
            trs.utrs.inserts_idx(num_background, new_rules)?;
            Ok(trs)
        } else {
            Err(SampleError::OptionsExhausted)
        }
    }
    /// returns a vector of a rules with each rhs being the lhs of the original
    /// rule and each lhs is each rhs of the original.
    fn swap_rule_helper(rule: &Rule) -> Result<Vec<Rule>, SampleError> {
        let rules = rule
            .clauses()
            .iter()
            .filter_map(TRS::swap_clause_helper)
            .collect_vec();
        if rules.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(rules)
        }
    }
    /// Swap lhs and rhs iff the rule is deterministic and swap is a valid rule.
    fn swap_clause_helper(rule: &Rule) -> Option<Rule> {
        rule.rhs()
            .and_then(|rhs| Rule::new(rhs, vec![rule.lhs.clone()]))
    }
}
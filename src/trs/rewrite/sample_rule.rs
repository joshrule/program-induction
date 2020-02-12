use super::{super::as_result, SampleError, TRS};
use rand::{seq::SliceRandom, Rng};
use std::collections::HashMap;
use term_rewriting::RuleContext;

impl<'a, 'b> TRS<'a, 'b> {
    /// Sample a rule and add it to the rewrite system.
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
    ///     parse_rule(&mut sig, "PLUS(x_ ZERO) = x_").expect("parsed rule"),
    ///     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))").expect("parsed rule"),
    /// ];
    ///
    /// let vars = vec![
    ///     ptp![int],
    ///     ptp![int],
    ///     ptp![int],
    /// ];
    /// let contexts = vec![
    ///     RuleContext {
    ///         lhs: Context::Hole,
    ///         rhs: vec![Context::Hole],
    ///     }
    /// ];
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, TypeContext::default());
    ///
    /// let mut trs = TRS::new(&lexicon, true, &[], rules).unwrap();
    ///
    /// assert_eq!(trs.len(), 2);
    ///
    /// let mut rng = thread_rng();
    /// let atom_weights = (1.0, 1.0, 1.0, 1.0);
    /// let max_size = 50;
    ///
    /// if let Ok(new_trss) = trs.sample_rule(&contexts, atom_weights, max_size, &mut rng) {
    ///     assert_eq!(new_trss[0].len(), 3);
    /// }
    /// ```
    pub fn sample_rule<R: Rng>(
        &self,
        contexts: &[RuleContext],
        atom_weights: (f64, f64, f64, f64),
        max_size: usize,
        rng: &mut R,
    ) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        let context = contexts
            .choose(rng)
            .ok_or(SampleError::OptionsExhausted)?
            .clone();
        let mut trs = self.clone();
        let rule = trs
            .lex
            .sample_rule_from_context(context, atom_weights, true, max_size, rng)
            .drop()?;
        if rule.lhs == rule.rhs().unwrap() {
            return Err(SampleError::Trivial);
        }
        trs.lex.infer_rule(&rule, &mut HashMap::new()).drop()?;
        let mut new_rules = vec![rule];
        trs.filter_background(&mut new_rules);
        // INVARIANT: there's at most one rule in new_rules
        let mut new_rules = as_result(new_rules)?;
        trs.utrs.push(new_rules.pop().unwrap())?;
        Ok(vec![trs])
    }
}

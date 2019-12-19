use super::{SampleError, TRS};
use rand::{seq::SliceRandom, Rng};
use std::collections::HashMap;

impl TRS {
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
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], contexts, false, TypeContext::default());
    ///
    /// let mut trs = TRS::new(&lexicon, rules).unwrap();
    ///
    /// assert_eq!(trs.len(), 2);
    ///
    /// let mut rng = thread_rng();
    /// let atom_weights = (1.0, 1.0, 1.0, 1.0);
    /// let max_size = 50;
    ///
    /// if let Ok(new_trss) = trs.sample_rule(atom_weights, max_size, &mut rng) {
    ///     assert_eq!(new_trss[0].len(), 3);
    /// }
    /// ```
    pub fn sample_rule<R: Rng>(
        &self,
        atom_weights: (f64, f64, f64, f64),
        max_size: usize,
        rng: &mut R,
    ) -> Result<Vec<TRS>, SampleError> {
        // TODO: fail if you sample an existing rule?
        let mut trs = self.clone();
        let context = {
            let contexts = &self.lex.0.read().expect("poisoned lexicon").templates;
            contexts
                .choose(rng)
                .ok_or(SampleError::OptionsExhausted)?
                .clone()
        };
        let rule = trs
            .lex
            .sample_rule_from_context(context, atom_weights, true, max_size)
            .drop()?;
        if rule.lhs == rule.rhs().unwrap() {
            return Err(SampleError::Trivial);
        }
        trs.lex.infer_rule(&rule, &mut HashMap::new()).drop()?;
        trs.utrs.push(rule)?;
        Ok(vec![trs])
    }
}

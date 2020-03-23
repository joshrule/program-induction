use super::{super::as_result, SampleError, TRS};
use polytype::TypeSchema;
use rand::Rng;
use std::collections::HashMap;

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
    /// # use programinduction::trs::{parse_lexicon, parse_trs};
    /// # use polytype::Context as TypeContext;
    /// # use rand::thread_rng;
    /// let mut lex = parse_lexicon(
    ///     "PLUS/2: int -> int -> int; SUCC/1: int -> int; ZERO/0: int;",
    ///     TypeContext::default(),
    /// )
    ///     .expect("parsed lexicon");
    ///
    /// let trs = parse_trs(
    ///     "PLUS(v0_ ZERO) = v0_; PLUS(v0_ SUCC(v1_)) = SUCC(PLUS(v0_ v1_));",
    ///     &mut lex,
    ///     false,
    ///     &[]
    /// )
    ///     .expect("parsed trs");
    ///
    /// let atom_weights = (1.0, 1.0, 1.0, 1.0);
    /// let max_size = 50;
    /// let mut rng = thread_rng();
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
    ) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        let mut trs = self.clone();
        let schema = TypeSchema::Monotype(trs.lex.0.to_mut().ctx.new_variable());
        let mut ctx = trs.lex.0.ctx.clone();
        let rule = trs
            .lex
            .sample_rule(&schema, atom_weights, max_size, true, &mut ctx, rng)?;
        if rule.lhs == rule.rhs().unwrap() {
            return Err(SampleError::Trivial);
        }
        trs.lex.infer_rule(&rule, &mut HashMap::new(), &mut ctx)?;
        let mut new_rules = vec![rule];
        trs.filter_background(&mut new_rules);
        // INVARIANT: there's at most one rule in new_rules
        let mut new_rules = as_result(new_rules)?;
        trs.utrs.push(new_rules.pop().unwrap())?;
        Ok(vec![trs])
    }
}

use polytype::atype::Variable as TVar;
use rand::Rng;
use std::collections::HashMap;
use trs::{as_result, Env, GenerationLimit, SampleError, SampleParams, TRS};

impl<'ctx, 'b> TRS<'ctx, 'b> {
    /// Sample a rule and add it to the rewrite system.
    ///
    /// # Example
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
    ///     let trs = parse_trs(
    ///         "PLUS(v0_ ZERO) = v0_; PLUS(v0_ SUCC(v1_)) = SUCC(PLUS(v0_ v1_));",
    ///         &mut lex, false, &[],
    ///     ).expect("trs");
    ///     let atom_weights = (1.0, 1.0, 1.0, 1.0);
    ///     let max_size = 50;
    ///
    ///     if let Ok(new_trss) = trs.sample_rule(atom_weights, max_size, &mut rng) {
    ///         assert_eq!(new_trss[0].len(), 3);
    ///     }
    /// })
    /// ```
    pub fn sample_rule<R: Rng>(
        &self,
        atom_weights: (f64, f64, f64, f64),
        max_size: usize,
        rng: &mut R,
    ) -> Result<Vec<Self>, SampleError<'ctx>> {
        let mut trs = self.clone();
        let tvar = TVar(trs.lex.lex.to_mut().src.fresh());
        let ctx = &trs.lex.lex.ctx;
        let schema = ctx.intern_monotype(ctx.intern_tvar(tvar));
        let params = SampleParams {
            atom_weights,
            limit: GenerationLimit::TermSize(max_size),
            variable: true,
        };
        let rule = trs.lex.sample_rule(&schema, params, true, rng)?;
        if rule.lhs == rule.rhs().unwrap() {
            return Err(SampleError::Trivial);
        }
        let mut new_rules = vec![rule];
        trs.filter_background(&mut new_rules);
        let new_rules = as_result(new_rules)?;
        trs.utrs.pushes(new_rules)?;
        Ok(vec![trs])
    }
}

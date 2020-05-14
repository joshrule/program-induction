use itertools::Itertools;
use polytype::{
    atype::{Schema, Snapshot as SubSnap, Substitution, Ty, Type, Variable as TVar},
    Source,
};
use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng,
};
use smallvec::{smallvec, SmallVec};
use std::convert::TryFrom;
use term_rewriting::{
    Atom, Context, Operator, PStringDist, Rule, RuleContext, SituatedAtom, Term, Variable,
    TRS as UntypedTRS,
};
use trs::{lexicon::GenerationLimit, Datum, Lexicon, SampleError, TypeError};

pub struct AtomEnumeration<'ctx, 'lex, 'atom> {
    tp: Ty<'ctx>,
    results: SmallVec<[(Schema<'ctx>, bool, bool); 32]>,
    env: &'atom mut Env<'ctx, 'lex>,
    op: usize,
    var: usize,
    invented: bool,
}

/// A Typing Environment.
#[derive(Debug)]
pub struct Env<'ctx, 'lex> {
    pub lex: Lexicon<'ctx, 'lex>,
    pub invent: bool,
    pub src: Source,
    pub vars: Vec<Schema<'ctx>>,
    pub tps: Vec<Ty<'ctx>>,
    pub sub: Substitution<'ctx>,
    pub(crate) fvs: Vec<TVar>,
}

//pub struct RuleEnumeration<'ctx, 'lex> {
//    stack: Vec<(RuleContext, (usize, usize), Vec<Ty<'ctx>>, Snapshot)>,
//    limit: GenerationLimit,
//    env: Env<'ctx, 'lex>,
//    invent: bool,
//}

#[derive(Copy, Clone, Debug)]
pub struct SampleParams {
    pub atom_weights: (f64, f64, f64, f64),
    pub variable: bool,
    pub limit: GenerationLimit,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Snapshot(bool, Source, usize, SubSnap, usize, usize);

//pub struct TermEnumeration<'ctx, 'lex> {
//    stack: Vec<(Context, usize, Vec<Ty<'ctx>>)>,
//    limit: GenerationLimit,
//    env: Env<'ctx, 'lex>,
//}

impl<'ctx, 'lex> Env<'ctx, 'lex> {
    pub fn new(invent: bool, lex: &Lexicon<'ctx, 'lex>, src: Option<Source>) -> Self {
        Env {
            lex: lex.clone(),
            invent,
            src: src.unwrap_or_else(|| Source::default()),
            vars: Vec::with_capacity(32),
            tps: Vec::with_capacity(32),
            sub: Substitution::with_capacity(lex.lex.ctx, 32),
            fvs: Vec::with_capacity(32),
        }
    }
    pub fn from_vars(
        vars: &[Variable],
        invent: bool,
        lex: &Lexicon<'ctx, 'lex>,
        src: Option<Source>,
    ) -> Self {
        let mut env = Env {
            lex: lex.clone(),
            invent,
            src: src.unwrap_or_else(|| Source::default()),
            vars: Vec::with_capacity(32),
            tps: Vec::with_capacity(32),
            sub: Substitution::with_capacity(lex.lex.ctx, 32),
            fvs: Vec::with_capacity(32),
        };
        env.add_variables(vars.len());
        env
    }
    pub fn snapshot(&self) -> Snapshot {
        Snapshot(
            self.invent,
            self.src,
            self.vars.len(),
            self.sub.snapshot(),
            self.tps.len(),
            self.fvs.len(),
        )
    }
    pub fn rollback(&mut self, Snapshot(i, _r, v, s, t, f): Snapshot) {
        self.invent = i;
        //self.src = r;
        self.vars.truncate(v);
        self.sub.rollback(s);
        self.tps.truncate(t);
        self.fvs.truncate(f);
    }
    pub fn add_variables(&mut self, n: usize) {
        for _ in 0..n {
            let tvar = TVar(self.src.fresh());
            let tp = self.lex.lex.ctx.intern_tvar(tvar);
            let ptp = self.lex.lex.ctx.intern_monotype(tp);
            self.vars.push(ptp);
            self.fvs.push(tvar)
        }
    }
    pub fn new_variable(&mut self) -> Option<Variable> {
        if self.invent {
            self.add_variables(1);
            Some(Variable(self.vars.len() - 1))
        } else {
            None
        }
    }
    pub fn free_vars(&self) -> Vec<TVar> {
        self.lex
            .lex
            .free_vars()
            .iter()
            .chain(&self.fvs)
            .flat_map(|x| self.lex.lex.ctx.intern_tvar(*x).apply(&self.sub).vars())
            .sorted()
            .unique()
            .collect_vec()
    }
    /// Infer the `TypeSchema` associated with an `Operator`.
    fn op_tp(&self, op: Operator) -> Result<(Schema<'ctx>, Option<usize>), TypeError<'ctx>> {
        let o_id = op.id();
        if o_id >= self.lex.lex.ops.len() {
            Err(TypeError::NotFound)
        } else {
            Ok((&self.lex.lex.ops[o_id].0, Some(self.lex.lex.ops[o_id].1)))
        }
    }
    /// Infer the `TypeSchema` associated with a `Variable`.
    fn var_tp(&self, var: Variable) -> Result<(Schema<'ctx>, Option<usize>), TypeError<'ctx>> {
        let v_id = var.id();
        if v_id >= self.vars.len() {
            Err(TypeError::NotFound)
        } else {
            Ok((&self.vars[v_id], None))
        }
    }
    /// Infer the [`TypeSchema`] associated with an [`Atom`].
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`Atom`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Atom.html
    pub fn infer_atom(
        &self,
        atom: &Atom,
    ) -> Result<(Schema<'ctx>, Option<usize>), TypeError<'ctx>> {
        match *atom {
            Atom::Operator(o) => self.op_tp(o),
            Atom::Variable(v) => self.var_tp(v),
        }
    }
    fn check_atom(&mut self, tp: Ty<'ctx>, atom: Atom) -> Result<Vec<Ty<'ctx>>, SampleError<'ctx>> {
        // println!("  tp: {}", tp);
        //for (i, v) in self.vars.iter().enumerate() {
        //    println!("  - {} *-> {}", i, v);
        //}
        //for (k, v) in self.sub.iter() {
        //    println!("  - {} |-> {}", k.0, v);
        //}
        let tp = tp.apply(&self.sub);
        // println!("  tp after apply: {}", tp);
        let (schema, constant) = match atom {
            Atom::Operator(op) => {
                let (schema, arity) = self.lex.lex.ops[op.id()];
                (schema, arity == 0)
            }
            Atom::Variable(var) => (self.vars[var.id()], true),
        };
        //println!("  schema : {}", schema);
        //println!("  constant : {}", constant);
        self.check_schema(&tp, schema, constant)
            .map_err(SampleError::from)
    }
    pub fn enumerate_atoms<'atom>(
        &'atom mut self,
        tp: Ty<'ctx>,
    ) -> AtomEnumeration<'ctx, 'lex, 'atom> {
        AtomEnumeration {
            tp: tp.apply(&self.sub),
            results: smallvec![],
            op: 0,
            var: 0,
            invented: false,
            env: self,
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn sample_atom<R: Rng>(
        &mut self,
        tp: Ty<'ctx>,
        params: SampleParams,
        rng: &mut R,
    ) -> Result<Atom, SampleError<'ctx>> {
        // List all the options.
        let mut options = self
            .enumerate_atoms(tp)
            .into_iter()
            .filter(|atom| match atom {
                Some(Atom::Variable(_)) | None => params.variable,
                Some(_) => true,
            })
            .collect_vec();
        if options.is_empty() {
            return Err(SampleError::OptionsExhausted);
        }
        // Weight the options.
        let weights = self.compute_option_weights(&options, params.atom_weights);
        // Sample an option.
        let indices = WeightedIndex::new(weights).unwrap();
        match options.swap_remove(indices.sample(rng)) {
            None => self
                .new_variable()
                .map(Atom::Variable)
                .ok_or(SampleError::Subterm),
            Some(atom) => Ok(atom),
        }
    }
    pub fn infer_term(&mut self, term: &Term) -> Result<Ty<'ctx>, TypeError<'ctx>> {
        match *term {
            Term::Variable(v) => {
                let tp = self
                    .var_tp(v)?
                    .0
                    .instantiate(&self.lex.lex.ctx, &mut self.src)
                    .apply(&self.sub);
                self.tps.push(tp);
                Ok(tp)
            }
            Term::Application { op, .. } if self.lex.lex.ops[op.id()].1 == 0 => {
                let tp = self
                    .op_tp(op)?
                    .0
                    .instantiate(&self.lex.lex.ctx, &mut self.src)
                    .apply(&self.sub);
                self.tps.push(tp);
                Ok(tp)
            }
            Term::Application { op, ref args } => {
                //println!("head_schema: {}/{:?}", self.op_tp(op)?.0, self.op_tp(op)?.1);
                let head_type = self
                    .op_tp(op)?
                    .0
                    .instantiate(&self.lex.lex.ctx, &mut self.src);
                //println!(
                //    "head type for {} {}",
                //    op.display(&self.lex.lex.sig),
                //    head_type
                //);
                let arg_types = head_type.args().unwrap();
                let return_tp = head_type.returns().unwrap_or(&head_type);
                //println!("return type {}", return_tp);
                self.tps.push(return_tp);
                for (a, arg_tp) in args.iter().zip(arg_types) {
                    let st_tp = self.infer_term(a)?;
                    //println!("unifying {} and {}", arg_tp, st_tp);
                    Type::unify_with_sub(&[(&arg_tp, &st_tp)], &mut self.sub)?;
                    //println!("success");
                }
                Ok(return_tp.apply(&self.sub))
            }
        }
    }
    // pub fn enumerate_terms(self, schema: Schema<'ctx>, n: usize) -> TermEnumeration<'ctx, 'lex> {
    //     TermEnumeration::new(self, schema, n)
    // }
    pub fn sample_term<R: Rng>(
        &mut self,
        context: &Context,
        arg_types: &mut Vec<Ty<'ctx>>,
        mut params: SampleParams,
        rng: &mut R,
    ) -> Result<Term, SampleError<'ctx>> {
        let mut size = [context.size()];
        if !params.limit.is_okay(&size) {
            return Err(SampleError::SizeExceeded);
        }
        let mut partial = context.clone();
        let variable = params.variable;
        loop {
            match Term::try_from(&partial) {
                Ok(term) => return Ok(term),
                Err(hole_place) => {
                    let tp = arg_types[0].apply(&self.sub);
                    let env_ss = self.snapshot();
                    params.variable = variable || partial.size() > 1;
                    let atom = self.sample_atom(&tp, params, rng)?;
                    let subcontext = Context::from(SituatedAtom::new(atom, &self.lex.lex.sig));
                    size[0] += subcontext.size() - 1;
                    if params.limit.is_okay(&size) {
                        let mut new_arg_types = self.check_atom(tp, atom)?;
                        partial = partial.replace(&hole_place, subcontext).unwrap();
                        new_arg_types.extend_from_slice(&arg_types[1..]);
                        *arg_types = new_arg_types;
                    } else {
                        self.rollback(env_ss);
                        return Err(SampleError::SizeExceeded);
                    }
                }
            }
        }
    }
    pub fn logprior_term(
        &mut self,
        term: &Term,
        tp: Ty<'ctx>,
        atom_weights: (f64, f64, f64, f64),
    ) -> Result<f64, SampleError<'ctx>> {
        let mut stack: SmallVec<[(&Term, Ty<'ctx>); 32]> = smallvec![(term, tp.apply(&self.sub))];
        let mut lp = 0.0;
        while let Some((term, tp)) = stack.pop() {
            // Update the type.
            tp.apply(&self.sub);
            // Collect all options.
            let option = self.compute_prior_weight(atom_weights, &tp, term.head());
            // Compute the probability of the term.
            match option {
                None => {
                    return Ok(std::f64::NEG_INFINITY);
                }
                Some((atom, w, z)) => {
                    lp += w.ln() - z.ln();
                    let arg_types = self.check_atom(tp, atom)?;
                    if let Term::Application { args, .. } = term {
                        stack.extend(args.iter().zip(arg_types).rev())
                    }
                }
            }
        }
        Ok(lp)
    }
    pub fn infer_rule(&mut self, rule: &Rule) -> Result<Ty<'ctx>, TypeError<'ctx>> {
        let lhs_type = self.infer_term(&rule.lhs)?;
        for rhs in &rule.rhs {
            let rhs_type = self.infer_term(&rhs)?;
            Type::unify_with_sub(&[(&rhs_type, &lhs_type)], &mut self.sub)?;
        }
        Ok(lhs_type)
    }
    //pub fn enumerate_rules(
    //    self,
    //    schema: Schema<'ctx>,
    //    n: GenerationLimit,
    //    invent: bool,
    //) -> RuleEnumeration<'ctx, 'lex> {
    //    RuleEnumeration::new(self, schema, n, invent)
    //}
    pub fn sample_rule<R: Rng>(
        &mut self,
        context: &RuleContext,
        arg_types: &mut Vec<Ty<'ctx>>,
        mut params: SampleParams,
        rng: &mut R,
    ) -> Result<Rule, SampleError<'ctx>> {
        let mut size = std::iter::once(context.lhs.size())
            .chain(context.rhs.iter().map(|rhs| rhs.size()))
            .collect_vec();
        if !params.limit.is_okay(&size) {
            return Err(SampleError::SizeExceeded);
        }
        let invent = self.invent;
        let variable = params.variable;
        let mut partial = context.clone();
        loop {
            match Rule::try_from(&partial) {
                Ok(rule) => return Ok(rule),
                Err(hole_place) => {
                    let lhs_hole = hole_place[0] == 0;
                    self.invent = invent && lhs_hole && hole_place != [0];
                    let tp = arg_types[0].apply(&self.sub);
                    params.variable = hole_place != [0] && variable;
                    let env_ss = self.snapshot();
                    let atom = self.sample_atom(tp, params, rng)?;
                    let subcontext = Context::from(SituatedAtom::new(atom, &self.lex.lex.sig));
                    size[hole_place[0]] += subcontext.size() - 1;
                    if params.limit.is_okay(&size) {
                        let mut new_arg_types = self.check_atom(tp, atom)?;
                        partial = partial.replace(&hole_place, subcontext).unwrap();
                        new_arg_types.extend_from_slice(&arg_types[1..]);
                        *arg_types = new_arg_types;
                    } else {
                        self.rollback(env_ss);
                        return Err(SampleError::SizeExceeded);
                    }
                }
            }
        }
    }
    /// Give the log probability of sampling a `Rule`.
    pub fn logprior_rule(
        &mut self,
        rule: &Rule,
        tp: Ty<'ctx>,
        atom_weights: (f64, f64, f64, f64),
    ) -> Result<f64, SampleError<'ctx>> {
        let lp_lhs = self.logprior_term(&rule.lhs, &tp, atom_weights)?;
        self.invent = false;
        let mut lp = 0.0;
        for rhs in &rule.rhs {
            lp += lp_lhs + self.logprior_term(&rhs, &tp, atom_weights)?;
        }
        Ok(lp)
    }
    pub fn infer_data(&mut self, data: &[Datum]) -> Result<Ty<'ctx>, TypeError<'ctx>> {
        let data_tps = data
            .iter()
            .map(|datum| {
                let ss = self.snapshot();
                let result = match datum {
                    Datum::Full(rule) => self.infer_rule(rule),
                    Datum::Partial(term) => self.infer_term(term),
                };
                self.rollback(ss);
                result
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tp = self.lex.lex.ctx.intern_tvar(TVar(self.src.fresh()));
        for rule_tp in data_tps {
            Type::unify_with_sub(&[(&tp, &rule_tp)], &mut self.sub)?;
        }
        Ok(tp.apply(&self.sub))
    }
    pub fn infer_context(&mut self, context: &Context) -> Result<Ty<'ctx>, TypeError<'ctx>> {
        match *context {
            Context::Hole => Ok(self.lex.lex.ctx.intern_tvar(TVar(self.src.fresh()))),
            Context::Variable(v) => {
                let tp = self
                    .var_tp(v)?
                    .0
                    .instantiate(&self.lex.lex.ctx, &mut self.src);
                Ok(tp)
            }
            Context::Application { op, .. } if self.lex.lex.ops[op.id()].1 == 0 => {
                let tp = self
                    .op_tp(op)?
                    .0
                    .instantiate(&self.lex.lex.ctx, &mut self.src)
                    .apply(&self.sub);
                Ok(tp)
            }
            Context::Application { op, ref args } => {
                let head_type = self
                    .op_tp(op)?
                    .0
                    .instantiate(&self.lex.lex.ctx, &mut self.src);
                let arg_types = head_type.args().unwrap();
                let return_tp = head_type.returns().unwrap_or(&head_type);
                self.tps.push(return_tp);
                for (a, arg_tp) in args.iter().zip(arg_types) {
                    let st_tp = self.infer_context(a)?;
                    Type::unify_with_sub(&[(&arg_tp, &st_tp)], &mut self.sub)?;
                }
                Ok(return_tp.apply(&self.sub))
            }
        }
    }
    pub fn infer_rulecontext(
        &mut self,
        context: &RuleContext,
    ) -> Result<Ty<'ctx>, TypeError<'ctx>> {
        let lhs_type = self.infer_context(&context.lhs)?;
        for rhs in &context.rhs {
            let rhs_type = self.infer_context(&rhs)?;
            Type::unify_with_sub(&[(&rhs_type, &lhs_type)], &mut self.sub)?;
        }
        Ok(lhs_type)
    }
    pub fn infer_utrs(&mut self, utrs: &UntypedTRS) -> Result<(), TypeError<'ctx>> {
        for rule in &utrs.rules {
            let ss = self.snapshot();
            self.add_variables(rule.variables().len().saturating_sub(self.vars.len()));
            self.infer_rule(rule)?;
            self.rollback(ss);
        }
        Ok(())
    }
    pub fn logprior_utrs<F>(
        &mut self,
        utrs: &UntypedTRS,
        p_of_n_rules: F,
        atom_weights: (f64, f64, f64, f64),
    ) -> Result<f64, SampleError<'ctx>>
    where
        F: Fn(usize) -> f64,
    {
        let tp = self.lex.lex.ctx.intern_tvar(TVar(self.src.fresh()));
        let mut p_rules = 0.0;
        for rule in &utrs.rules {
            let ss = self.snapshot();
            p_rules += self.logprior_rule(rule, tp, atom_weights)?;
            self.rollback(ss);
        }
        Ok(p_of_n_rules(utrs.clauses().len()) + p_rules)
    }
    pub fn logprior_srs<F>(
        &mut self,
        srs: &UntypedTRS,
        p_of_n_rules: F,
        atom_weights: (f64, f64, f64, f64),
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
    ) -> Result<f64, SampleError<'ctx>>
    where
        F: Fn(usize) -> f64,
    {
        let tp = self.lex.lex.ctx.intern_tvar(TVar(self.src.fresh()));
        let mut p_clauses = 0.0;
        for rule in &srs.rules {
            p_clauses += self.logprior_string_rule(rule, tp, atom_weights, dist, t_max, d_max)?;
        }
        Ok(p_of_n_rules(srs.clauses().len()) + p_clauses)
    }
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::too_many_arguments))]
    fn logprior_string_rule(
        &mut self,
        rule: &Rule,
        tp: Ty<'ctx>,
        atom_weights: (f64, f64, f64, f64),
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
    ) -> Result<f64, SampleError<'ctx>> {
        let lp_lhs = self.logprior_term(&rule.lhs, tp, atom_weights)?;
        let mut lp = 0.0;
        for rhs in &rule.rhs {
            lp += lp_lhs
                + UntypedTRS::p_string(&rule.lhs, rhs, dist, t_max, d_max, &self.lex.lex.sig)
                    .ok_or(SampleError::Subterm)?
        }
        Ok(lp)
    }
    fn compute_option_weights(
        &self,
        options: &[Option<Atom>],
        (vw, cw, ow, iw): (f64, f64, f64, f64),
    ) -> Vec<f64> {
        options
            .iter()
            .map(|atom| match atom {
                Some(Atom::Operator(o)) if o.arity(&self.lex.lex.sig) == 0 => cw,
                Some(Atom::Operator(_)) => ow,
                Some(Atom::Variable(v)) if self.vars.len() > v.id() => vw,
                None if self.invent => iw,
                _ => 0.0,
            })
            .collect()
    }
    fn compute_prior_weight(
        &mut self,
        atom_weights: (f64, f64, f64, f64),
        tp: Ty<'ctx>,
        head: Atom,
    ) -> Option<(Atom, f64, f64)> {
        let mut options = self.enumerate_atoms(tp).collect_vec();
        let mut weights = self.compute_option_weights(&options, atom_weights);
        let z: f64 = weights.iter().sum();
        let invented = self.invent
            && head.is_variable()
            && !options.iter().any(|atom| match atom {
                Some(a) => *a == head,
                None => false,
            });
        let idx = options.iter().position(|atom| match atom {
            Some(a) => *a == head,
            None => invented,
        });
        idx.map(|idx| {
            (
                options
                    .swap_remove(idx)
                    .or_else(|| self.new_variable().map(Atom::Variable))
                    .expect("invented variable"),
                weights.swap_remove(idx),
                z,
            )
        })
    }
    fn check_schema(
        &mut self,
        tp: Ty<'ctx>,
        schema: Schema<'ctx>,
        constant: bool,
    ) -> Result<Vec<Ty<'ctx>>, TypeError<'ctx>> {
        let query_tp = schema.instantiate(&self.lex.lex.ctx, &mut self.src);
        // println!("    checking {} against {}", tp, query_tp);
        let result = if constant {
            Type::unify_with_sub(&[(&query_tp, &tp)], &mut self.sub).map(|_| Vec::new())?
        } else {
            Type::unify_with_sub(
                &[(query_tp.returns().ok_or(TypeError::Malformed)?, tp)],
                &mut self.sub,
            )
            .map(|_| query_tp.args().unwrap_or_else(Vec::new))?
        };
        Ok(result)
    }
}

impl<'ctx, 'lex, 'atom> Iterator for AtomEnumeration<'ctx, 'lex, 'atom> {
    type Item = Option<Atom>;
    fn next(&mut self) -> Option<Self::Item> {
        while self.op < self.env.lex.lex.ops.len() {
            let (schema, arity): (Schema<'ctx>, usize) = self.env.lex.lex.ops[self.op];
            let constant = arity == 0;
            self.op += 1;
            match self
                .results
                .iter()
                .find(|(s, c, _)| *s == schema && *c == constant)
                .map(|(_, _, r)| r)
            {
                None => {
                    let snapshot = self.env.snapshot();
                    let fit = self.env.check_schema(self.tp, schema, constant).is_ok();
                    self.env.rollback(snapshot);
                    self.results.push((&schema, constant, fit));
                    if fit {
                        return Some(Some(Atom::Operator(Operator(self.op - 1))));
                    }
                }
                Some(true) => {
                    return Some(Some(Atom::Operator(Operator(self.op - 1))));
                }
                _ => (),
            }
        }
        while self.var < self.env.vars.len() {
            let schema = self.env.vars[self.var];
            self.var += 1;
            let snapshot = self.env.snapshot();
            if self.env.check_schema(self.tp, schema, true).is_ok() {
                self.env.rollback(snapshot);
                return Some(Some(Atom::Variable(Variable(self.var - 1))));
            }
            self.env.rollback(snapshot);
        }
        if self.env.invent && !self.invented {
            self.invented = true;
            return Some(None);
        }
        None
    }
}

// impl<'ctx, 'lex> TermEnumeration<'ctx, 'lex> {
//     fn new(mut env: Env<'ctx, 'lex>, schema: Schema<'ctx>, max_size: usize) -> Self {
//         let tp = schema.instantiate(&env.lex.lex.ctx, &mut env.src);
//         let limit = GenerationLimit::TotalSize(max_size);
//         TermEnumeration {
//             stack: vec![(Context::Hole, 1, vec![tp])],
//             limit,
//             env,
//         }
//     }
// }

// impl<'ctx, 'lex> Iterator for TermEnumeration<'ctx, 'lex> {
//     type Item = Term;
//     fn next(&mut self) -> Option<Self::Item> {
//         while let Some((partial, size, arg_types)) = self.stack.pop() {
//             match Term::try_from(&partial) {
//                 Ok(term) => return Some(term),
//                 Err(hole_place) => {
//                     let ss = self.env.snapshot();
//                     for v in partial.variables() {
//                         if v.id() > self.env.vars.len() {
//                             self.env.new_variable();
//                         }
//                     }
//                     // Find the options fitting the hole:
//                     // TODO: annoying that we allocate here. Is there a better way?
//                     for opt_atom in self.env.enumerate_atoms(&arg_types[0]).collect_vec() {
//                         let ss = self.env.snapshot();
//                         let atom = opt_atom
//                             .or_else(|| self.env.new_variable().map(Atom::Variable))
//                             .unwrap();
//                         let subcontext =
//                             Context::from(SituatedAtom::new(atom, &self.env.lex.lex.sig));
//                         let new_size = size + subcontext.size() - 1;
//                         if self.limit.is_okay(&[new_size]) {
//                             if let Ok(mut new_arg_types) = self.env.check_atom(&arg_types[0], atom)
//                             {
//                                 let new_context = partial.replace(&hole_place, subcontext).unwrap();
//                                 new_arg_types.extend_from_slice(&arg_types[1..]);
//                                 self.stack.push((new_context, new_size, new_arg_types));
//                             }
//                         }
//                         self.env.rollback(ss);
//                     }
//                     self.env.rollback(ss);
//                 }
//             }
//         }
//         None
//     }
// }

// impl<'ctx, 'lex> RuleEnumeration<'ctx, 'lex> {
//     fn new(
//         mut env: Env<'ctx, 'lex>,
//         schema: Schema<'ctx>,
//         limit: GenerationLimit,
//         invent: bool,
//     ) -> Self {
//         let tp = schema.instantiate(&env.lex.lex.ctx, &mut env.src);
//         RuleEnumeration {
//             stack: vec![(RuleContext::default(), (1, 1), vec![tp, tp], env.snapshot())],
//             limit,
//             env,
//             invent,
//         }
//     }
// }

// impl<'ctx, 'lex> Iterator for RuleEnumeration<'ctx, 'lex> {
//     type Item = Rule;
//     fn next(&mut self) -> Option<Self::Item> {
//         while let Some((partial, (lsize, rsize), arg_types, ss)) = self.stack.pop() {
//             println!("looking at {}", partial.pretty(&self.env.lex.lex.sig));
//             match Rule::try_from(&partial) {
//                 Ok(rule) => return Some(rule),
//                 Err(hole_place) => {
//                     self.env.rollback(ss);
//                     println!("hole_place {:?}", hole_place);
//                     self.env.add_variables(
//                         partial
//                             .variables()
//                             .len()
//                             .saturating_sub(self.env.vars.len()),
//                     );
//                     if self.env.infer_rulecontext(&partial).is_err() {
//                         continue;
//                     }
//                     let ss = self.env.snapshot();
//                     let lhs_hole = hole_place[0] == 0;
//                     self.env.invent = self.invent && lhs_hole && hole_place != [0];
//                     println!("self.env.invent: {}", self.env.invent);
//                     println!("arg types");
//                     for arg in &arg_types {
//                         println!("- {}", arg);
//                     }
//                     for opt_atom in self
//                         .env
//                         .enumerate_atoms(arg_types[0].apply(&self.env.sub))
//                         .collect_vec()
//                     {
//                         self.env.rollback(ss);
//                         println!("considering {:?}", opt_atom);
//                         let atom = opt_atom
//                             .or_else(|| self.env.new_variable().map(Atom::Variable))
//                             .unwrap();
//                         println!("that is {}", atom.display(&self.env.lex.lex.sig));
//                         let subcontext =
//                             Context::from(SituatedAtom::new(atom, &self.env.lex.lex.sig));
//                         println!("that is {}", subcontext.pretty(&self.env.lex.lex.sig));
//                         let new_size = if lhs_hole {
//                             (lsize + subcontext.size() - 1, rsize)
//                         } else {
//                             (lsize, rsize + subcontext.size() - 1)
//                         };
//                         println!("new_size {:?}", new_size);
//                         if self.limit.is_okay(&[new_size.0, new_size.1]) {
//                             println!("size is okay");
//                             if let Ok(mut new_arg_types) = self.env.check_atom(&arg_types[0], atom)
//                             {
//                                 println!("atom checks out against {}", &arg_types[0]);
//                                 let new_context = partial.replace(&hole_place, subcontext).unwrap();
//                                 println!(
//                                     "new_context {}",
//                                     new_context.pretty(&self.env.lex.lex.sig)
//                                 );
//                                 new_arg_types.extend_from_slice(&arg_types[1..]);
//                                 for na in &new_arg_types {
//                                     println!("  >> {}", na);
//                                 }
//                                 self.stack.push((new_context, new_size, new_arg_types, ss));
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//         None
//     }
// }

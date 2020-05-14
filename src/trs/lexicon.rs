use itertools::Itertools;
use polytype::{
    atype::{Schema, Ty, TypeContext, Variable as TypeVar},
    Source,
};
use rand::Rng;
use std::borrow::Cow;
use std::{collections::HashMap, fmt};
use term_rewriting::{
    Atom, Context, MergeStrategy, Operator, PStringDist, Rule, RuleContext, Signature,
    SignatureChange, Term, TRS as UntypedTRS,
};
use trs::{Datum, Env, SampleError, SampleParams, TypeError};

#[derive(Copy, Clone, Debug)]
pub enum GenerationLimit {
    TotalSize(usize),
    TermSize(usize),
}

#[derive(Clone, Debug)]
pub struct Lexicon<'ctx, 'lex> {
    pub lex: Cow<'lex, Lex<'ctx>>,
}

/// Manages the syntax of a term rewriting system.
#[derive(Clone, Debug)]
pub struct Lex<'ctx> {
    pub sig: Signature,
    pub src: Source,
    pub ctx: TypeContext<'ctx>,
    pub(crate) ops: Vec<(Schema<'ctx>, usize)>,
    pub(crate) tps: HashMap<Schema<'ctx>, Vec<Operator>>,
    fvs: Vec<TypeVar>,
}

impl GenerationLimit {
    /// Test whether the size of a generated item matches the `GenerationLimit`.
    pub fn is_okay(&self, size: &[usize]) -> bool {
        match *self {
            GenerationLimit::TotalSize(n) => n >= size.iter().sum(),
            GenerationLimit::TermSize(n) => n >= *size.iter().max().unwrap(),
        }
    }
}

impl<'ctx, 'lex> Lexicon<'ctx, 'lex> {
    /// Convert a [`term_rewriting::Signature`] into a `Lexicon`. `ops` are
    /// types for the [`term_rewriting::Operator`]s.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::Lexicon;
    /// # use term_rewriting::Signature;
    /// # use polytype::{Source, atype::{with_ctx, TypeSchema, TypeContext}};
    /// with_ctx(10, |ctx: TypeContext<'_>| {
    ///     let mut sig = Signature::default();
    ///     sig.new_op(2, Some("PLUS".to_string()));
    ///     sig.new_op(1, Some("SUCC".to_string()));
    ///     sig.new_op(0, Some("ZERO".to_string()));
    ///
    ///     let t_plus = TypeSchema::parse(&ctx, "int -> int -> int").expect("t_plus");
    ///     let t_succ = TypeSchema::parse(&ctx, "int -> int").expect("t_succ");
    ///     let t_zero = TypeSchema::parse(&ctx, "int").expect("t_zero");
    ///     let ops = vec![t_plus, t_succ, t_zero];
    ///
    ///     let src = Source::default();
    ///
    ///     let lexicon = Lexicon::from_signature(sig, ops, ctx, src);
    /// })
    /// ```
    ///
    /// [`polytype::ptp`]: https://docs.rs/polytype/~6.0/polytype/macro.ptp.html
    /// [`polytype::TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`term_rewriting::Signature`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Signature.html
    /// [`term_rewriting::Operator`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Operator.html
    pub fn from_signature(
        sig: Signature,
        ops: Vec<Schema<'ctx>>,
        ctx: TypeContext<'ctx>,
        src: Source,
    ) -> Self {
        let ops = ops
            .into_iter()
            .enumerate()
            .map(|(i, t)| (t, Operator(i).arity(&sig) as usize))
            .collect_vec();
        let mut lex = Lex {
            ops,
            sig,
            src,
            fvs: vec![],
            tps: HashMap::new(),
            ctx,
        };
        lex.recompute_free_vars();
        lex.recompute_types();
        Lexicon {
            lex: Cow::Owned(lex),
        }
    }
    /// Return the `Lexicon`'s `Signature`.
    pub fn signature(&self) -> &Signature {
        &self.lex.sig
    }
    /// Return the `Lexicon`'s `Source`.
    pub fn source(&self) -> Source {
        self.lex.src
    }
    /// Return the specified `Operator` if possible.
    pub fn has_operator(
        &self,
        name: Option<&str>,
        arity: u8,
    ) -> Result<Operator, SampleError<'ctx>> {
        self.lex
            .sig
            .has_op(arity, name.map(String::from))
            .ok_or(SampleError::OptionsExhausted)
    }
    /// Merge two `Lexicon`s into a single `Lexicon`.
    pub fn merge(
        lex1: &Self,
        lex2: &Self,
        strategy: MergeStrategy,
    ) -> Result<(Self, SignatureChange), ()> {
        // 0. Check the contexts.
        if lex1.lex.ctx != lex2.lex.ctx {
            return Err(());
        }
        // 1. Merge the signatures.
        let mut sig = lex1.lex.sig.clone();
        let sig_change = sig.merge(&lex2.lex.sig.clone(), strategy)?;
        let mut inv_map = HashMap::new();
        for (k, v) in sig_change.op_map.iter() {
            inv_map.insert(v, k);
        }
        // 2. Merge the type contexts.
        let mut src = lex1.lex.src;
        let sacreds = lex2
            .lex
            .fvs
            .iter()
            .filter(|fv| lex1.lex.fvs.contains(fv))
            .cloned()
            .collect_vec();
        let change = src.merge(lex2.lex.src, sacreds);
        // 3. Update ops.
        let mut ops = lex1.lex.ops.clone();
        for op in &sig.operators() {
            let id = op.id();
            if id >= lex1.lex.ops.len() {
                let (mut schema, arity) = lex2.lex.ops[*inv_map[&id]].clone();
                change.reify_typeschema(&mut schema, &lex1.lex.ctx);
                ops.push((schema, arity));
            }
        }
        // 4. Create and return a new Lexicon from parts.
        let mut lex = Lex {
            ops,
            sig: sig,
            src,
            fvs: vec![],
            tps: HashMap::new(),
            ctx: lex1.lex.ctx,
        };
        lex.recompute_free_vars();
        lex.recompute_types();
        let lex = Lexicon {
            lex: Cow::Owned(lex),
        };
        Ok((lex, sig_change))
    }
    pub fn invent_operator(&mut self, name: Option<String>, arity: u8, tp: Ty<'ctx>) -> Operator {
        self.lex.to_mut().invent_operator(name, arity, tp)
    }
    pub fn free_vars(&self) -> Vec<TypeVar> {
        self.lex.free_vars()
    }
}

impl<'ctx, 'lex> PartialEq for Lexicon<'ctx, 'lex> {
    fn eq(&self, other: &Self) -> bool {
        self.lex == other.lex
    }
}

impl<'ctx, 'lex> Eq for Lexicon<'ctx, 'lex> {}

impl<'ctx, 'lex> fmt::Display for Lexicon<'ctx, 'lex> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.lex)
    }
}

impl<'ctx, 'lex> Lexicon<'ctx, 'lex> {
    /// Infer the [`TypeSchema`] associated with an [`Atom`].
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`Atom`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Atom.html
    pub fn infer_atom(
        &self,
        atom: &Atom,
    ) -> Result<(Schema<'ctx>, Option<usize>), TypeError<'ctx>> {
        let env = Env::from_vars(&atom.variables(), false, self, Some(self.lex.src));
        env.infer_atom(atom)
    }
    /// Infer the [`TypeSchema`] associated with a [`Term`].
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::{parse_term, parse_lexicon};
    /// # use polytype::atype::with_ctx;
    /// with_ctx(32, |ctx| {
    ///     let mut lex = parse_lexicon(
    ///        "PLUS/2: int -> int -> int; SUCC/1: int -> int; ZERO/0: int;",
    ///        &ctx,
    ///     ).expect("lex");
    ///
    ///     let term = parse_term("PLUS(v0_ SUCC(PLUS(SUCC(ZERO) SUCC(ZERO))))", &mut lex).expect("term");
    ///     let env = lex.infer_term(&term).expect("env");
    ///     let tp = env.tps[0].apply(&env.sub);
    ///     assert_eq!(tp.to_string(), "int")
    /// })
    /// ```
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`Term`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Term.html
    pub fn infer_term(&self, term: &Term) -> Result<Env<'ctx, 'lex>, TypeError<'ctx>> {
        let mut env = Env::from_vars(&term.variables(), false, self, Some(self.lex.src));
        env.infer_term(term)?;
        Ok(env)
    }
    ///// Enumerate `Term`s of length `n` or less.
    /////
    ///// # Examples
    /////
    ///// ```
    ///// # #[macro_use] extern crate polytype;
    ///// # extern crate programinduction;
    ///// # use programinduction::trs::{parse_lexicon};
    ///// # use polytype::atype::{Variable as TVar, TypeSchema, with_ctx};
    ///// with_ctx(32, |ctx| {
    /////     let mut lex = parse_lexicon("A/0: term; ./2: term -> term -> term;", &ctx).expect("lex");
    /////     let schema = TypeSchema::parse(&ctx, "term").expect("schema");
    /////     let terms = lex.enumerate_terms(&schema, false, 11);
    /////     assert_eq!(terms.count(), 65);
    ///// })
    ///// ```
    //pub fn enumerate_terms(
    //    &self,
    //    schema: Schema<'ctx>,
    //    invent: bool,
    //    n: usize,
    //) -> TermEnumeration<'ctx, 'lex> {
    //    let env = Env::new(invent, self, Some(self.lex.src));
    //    env.enumerate_terms(schema, n)
    //}
    /// Sample a [`term_rewriting::Term`].
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # use programinduction::trs::{SampleParams, GenerationLimit, parse_lexicon};
    /// # use polytype::atype::{Variable as TVar, TypeSchema, with_ctx};
    /// # use rand::thread_rng;
    /// with_ctx(32, |ctx| {
    ///     let mut lex = parse_lexicon("PLUS/2: int -> int -> int; SUCC/1: int -> int; ZERO/0: int;", &ctx).expect("lex");
    ///     let schema = TypeSchema::parse(&ctx, "int").expect("schema");
    ///     let params = SampleParams {
    ///         limit: GenerationLimit::TermSize(20),
    ///         variable: true,
    ///         atom_weights: (2.0, 2.0, 1.0, 2.0),
    ///     };
    ///     let mut rng = thread_rng();
    ///     for i in 0..50 {
    ///         let term = lex.sample_term(&schema, params, true, &mut rng).unwrap();
    ///         println!("{}. {}", i, term.pretty(&lex.signature()));
    ///     }
    /// })
    /// ```
    ///
    /// [`term_rewriting::Term`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Term.html
    pub fn sample_term<R: Rng>(
        &self,
        schema: Schema<'ctx>,
        params: SampleParams,
        invent: bool,
        rng: &mut R,
    ) -> Result<Term, SampleError<'ctx>> {
        let mut env = Env::new(invent, self, Some(self.lex.src));
        let partial = Context::Hole;
        let tp = schema
            .instantiate(&self.lex.ctx, &mut env.src)
            .apply(&env.sub);
        let mut arg_types = vec![tp];
        env.sample_term(&partial, &mut arg_types, params, rng)
    }
    /// Sample a `Term` conditioned on a `Context` rather than a `TypeSchema`.
    pub fn sample_term_from_context<R: Rng>(
        &self,
        context: &Context,
        params: SampleParams,
        invent: bool,
        rng: &mut R,
    ) -> Result<Term, SampleError<'ctx>> {
        let mut env = Env::new(invent, self, Some(self.lex.src));
        env.add_variables(context.variables().len());
        env.infer_context(&context)?;
        let mut arg_types = context
            .subcontexts()
            .iter()
            .zip(&env.tps)
            .filter(|((c, _), _)| c.is_hole())
            .map(|((_, _), t)| *t)
            .collect_vec();
        env.sample_term(context, &mut arg_types, params, rng)
    }
    /// Give the log probability of sampling a `Term`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::{parse_lexicon, parse_term};
    /// # use polytype::atype::{Variable as TVar, TypeSchema, with_ctx};
    /// with_ctx(32, |ctx| {
    ///     let mut lex = parse_lexicon("A/0: term; ./2: term -> term -> term;", &ctx).expect("lex");
    ///     let term = parse_term("(A ((v0_ v1_) (v1_ (A v0_))))", &mut lex).expect("parsed term");
    ///     let schema = ctx.intern_monotype(ctx.intern_tvar(TVar(lex.lex.to_mut().src.fresh())));
    ///     let atom_weights = (1.0, 1.0, 1.0, 1.0);
    ///     let lp = lex.logprior_term(&term, &schema, true, atom_weights).unwrap();
    ///     assert_eq!((lp*1e9_f64).round(), ((-5.0*3.0_f64.ln() - 4.0_f64.ln() - 5.0*5.0_f64.ln())*1e9_f64).round());
    /// })
    /// ```
    pub fn logprior_term(
        &self,
        term: &Term,
        schema: Schema<'ctx>,
        invent: bool,
        atom_weights: (f64, f64, f64, f64),
    ) -> Result<f64, SampleError<'ctx>> {
        let mut env = Env::new(invent, self, Some(self.lex.src));
        let tp = schema.instantiate(&env.lex.lex.ctx, &mut env.src);
        env.logprior_term(term, tp, atom_weights)
    }
    /// Infer the [`TypeSchema`] associated with a [`Rule`].
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
    pub fn infer_rule(&self, rule: &Rule) -> Result<Env<'ctx, 'lex>, TypeError<'ctx>> {
        let mut env = Env::from_vars(&rule.variables(), false, self, Some(self.lex.src));
        env.infer_rule(rule)?;
        Ok(env)
    }
    ///// Enumerate [`Rule`]s of length `n` or less.
    /////
    ///// # Examples
    /////
    ///// ```
    ///// # #[macro_use] extern crate polytype;
    ///// # extern crate programinduction;
    ///// # use programinduction::trs::{GenerationLimit, parse_lexicon};
    ///// # use polytype::atype::{Variable as TVar, TypeSchema, with_ctx};
    ///// with_ctx(32, |ctx| {
    /////     let mut lex = parse_lexicon("A/0: term; ./2: term -> term -> term;", &ctx).expect("lex");
    /////     let schema = TypeSchema::parse(&ctx, "term").expect("schema");
    /////     let rules = lex.enumerate_rules(&schema, GenerationLimit::TotalSize(11), false);
    /////     assert_eq!(rules.count(), 64);
    ///// })
    ///// ```
    /////
    ///// ```
    ///// # #[macro_use] extern crate polytype;
    ///// # extern crate programinduction;
    ///// # use programinduction::trs::{GenerationLimit, parse_lexicon};
    ///// # use polytype::atype::{Variable as TVar, TypeSchema, with_ctx};
    ///// with_ctx(32, |ctx| {
    /////     let mut lex = parse_lexicon("id/0: t1. t1 -> t1; ./2: t1. t2. (t1 -> t2) -> t1 -> t2;", &ctx).expect("lex");
    /////     println!("lex:\n{}", lex);
    /////     let schema = ctx.intern_monotype(ctx.intern_tvar(TVar(lex.lex.to_mut().src.fresh())));
    /////     println!("schema: {}", schema);
    /////     let rules: Vec<_> = lex.enumerate_rules(&schema, GenerationLimit::TermSize(3), true).collect();
    /////     for rule in &rules {
    /////         println!("{}", rule.pretty(&lex.lex.sig));
    /////     }
    /////     assert_eq!(rules.len(), 18);
    ///// })
    ///// ```
    /////
    ///// [`Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
    //pub fn enumerate_rules(
    //    &self,
    //    schema: Schema<'ctx>,
    //    n: GenerationLimit,
    //    invent: bool,
    //) -> RuleEnumeration<'ctx, 'lex> {
    //    let env = Env::new(invent, self, Some(self.lex.src));
    //    env.enumerate_rules(schema, n, invent)
    //}
    /// Sample a `Rule`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # use programinduction::trs::{SampleParams, GenerationLimit, parse_lexicon};
    /// # use polytype::atype::{Variable as TVar, TypeSchema, with_ctx};
    /// # use rand::thread_rng;
    /// with_ctx(32, |ctx| {
    ///     let mut lex = parse_lexicon("PLUS/2: int -> int -> int; SUCC/1: int -> int; ZERO/0: int;", &ctx).expect("lex");
    ///     let schema = TypeSchema::parse(&ctx, "int").expect("schema");
    ///     let params = SampleParams {
    ///         limit: GenerationLimit::TotalSize(40),
    ///         variable: true,
    ///         atom_weights: (3.0, 3.0, 1.0, 3.0),
    ///     };
    ///     let mut rng = thread_rng();
    ///     for i in 0..50 {
    ///         let rule = lex.sample_rule(&schema, params, true, &mut rng).unwrap();
    ///         println!("{}. {}", i, rule.pretty(&lex.signature()));
    ///     }
    /// })
    /// ```
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # use programinduction::trs::{SampleParams, GenerationLimit, parse_lexicon};
    /// # use polytype::atype::{Variable as TVar, TypeSchema, with_ctx};
    /// # use rand::thread_rng;
    /// with_ctx(32, |ctx| {
    ///     let mut lex = parse_lexicon(
    ///         "equal/0: t1. t1 -> t1 -> bool; >/0: int -> int -> bool;",
    ///         &ctx,
    ///     ).expect("lex");
    ///     let schema = ctx.intern_monotype(ctx.intern_tvar(TVar(lex.lex.to_mut().src.fresh())));
    ///     let params = SampleParams {
    ///         limit: GenerationLimit::TermSize(1),
    ///         variable: true,
    ///         atom_weights: (3.0, 3.0, 1.0, 3.0),
    ///     };
    ///     let mut rng = thread_rng();
    ///
    ///     let mut rules = vec![];
    ///     while rules.len() < 50 {
    ///         if let Ok(rule) = lex.sample_rule(&schema, params, false, &mut rng) {
    ///             rules.push(rule.pretty(&lex.signature()));
    ///             println!("{}", rule.pretty(&lex.signature()));
    ///         }
    ///     }
    ///     assert!(rules.contains(&"> = equal".to_string()));
    ///     assert!(rules.contains(&"equal = >".to_string()));
    /// })
    /// ```
    pub fn sample_rule<R: Rng>(
        &self,
        schema: Schema<'ctx>,
        params: SampleParams,
        invent: bool,
        rng: &mut R,
    ) -> Result<Rule, SampleError<'ctx>> {
        let mut env = Env::new(invent, self, Some(self.lex.src));
        let partial = RuleContext::default();
        let tp = schema
            .instantiate(&self.lex.ctx, &mut env.src)
            .apply(&env.sub);
        let mut arg_types = vec![tp, tp];
        env.sample_rule(&partial, &mut arg_types, params, rng)
    }
    /// Sample a `Rule` conditioned on a `Context` rather than a `TypeSchema`.
    pub fn sample_rule_from_context<R: Rng>(
        &self,
        context: &RuleContext,
        params: SampleParams,
        invent: bool,
        rng: &mut R,
    ) -> Result<Rule, SampleError<'ctx>> {
        let mut env = Env::new(invent, self, Some(self.lex.src));
        env.add_variables(context.variables().len());
        env.infer_rulecontext(&context)?;
        let mut arg_types = context
            .subcontexts()
            .iter()
            .zip(&env.tps)
            .filter(|((c, _), _)| c.is_hole())
            .map(|((_, _), t)| *t)
            .collect_vec();
        env.sample_rule(&context, &mut arg_types, params, rng)
    }
    /// Give the log probability of sampling a `Rule`.
    pub fn logprior_rule(
        &self,
        rule: &Rule,
        schema: Schema<'ctx>,
        invent: bool,
        atom_weights: (f64, f64, f64, f64),
    ) -> Result<f64, SampleError<'ctx>> {
        let mut env = Env::new(invent, self, Some(self.lex.src));
        let tp = schema.instantiate(&env.lex.lex.ctx, &mut env.src);
        env.logprior_rule(rule, tp, atom_weights)
    }
    /// Infer the [`TypeSchema`] associated with a slice of [`Rule`]s.
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
    // TODO: write test for infer_data - is this correct?
    pub fn infer_data(&self, data: &[Datum]) -> Result<Env<'ctx, 'lex>, TypeError<'ctx>> {
        let mut env = Env::new(true, self, Some(self.lex.src));
        env.infer_data(data)?;
        Ok(env)
    }
    /// Infer the [`TypeSchema`] associated with a [`Context`].
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::{parse_context, parse_lexicon};
    /// # use polytype::atype::with_ctx;
    /// with_ctx(32, |ctx| {
    ///     let mut lex = parse_lexicon(
    ///        "PLUS/2: int -> int -> int; SUCC/1: int -> int; ZERO/0: int;",
    ///        &ctx,
    ///     ).expect("lex");
    ///
    ///     let context = parse_context("SUCC([!])", &mut lex).expect("context");
    ///     let env = lex.infer_context(&context).expect("env");
    ///     let tp = env.tps[0].apply(&env.sub);
    ///     assert_eq!(tp.to_string(), "int")
    /// })
    /// ```
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`Context`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Context.html
    pub fn infer_context(&self, context: &Context) -> Result<Env<'ctx, 'lex>, TypeError<'ctx>> {
        let mut env = Env::from_vars(&context.variables(), false, self, Some(self.lex.src));
        env.infer_context(context)?;
        Ok(env)
    }
    /// Infer the [`TypeSchema`] associated with a [`RuleContext`].
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`RuleContext`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.RuleContext.html
    pub fn infer_rulecontext<'b>(
        &self,
        context: &RuleContext,
    ) -> Result<Env<'ctx, 'lex>, TypeError<'ctx>> {
        let mut env = Env::from_vars(&context.variables(), false, self, Some(self.lex.src));
        env.infer_rulecontext(context)?;
        Ok(env)
    }
    /// Infer the [`TypeSchema`] associated with a [`TRS`].
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`RuleContext`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.RuleContext.html
    pub fn infer_utrs(&self, utrs: &UntypedTRS) -> Result<Env<'ctx, 'lex>, TypeError<'ctx>> {
        let mut env = Env::new(true, self, Some(self.lex.src));
        env.infer_utrs(utrs)?;
        Ok(env)
    }
    /// Give the log probability of sampling a TRS.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::{parse_lexicon, parse_trs};
    /// # use polytype::atype::{Variable as TVar, TypeSchema, with_ctx};
    /// with_ctx(32, |ctx| {
    ///     let mut lex = parse_lexicon(
    ///         &[
    ///             "C/0: list -> list;",
    ///             "CONS/0: nat -> list -> list;",
    ///             "NIL/0: list;",
    ///             "HEAD/0: list -> nat;",
    ///             "TAIL/0: list -> list;",
    ///             "ISEMPTY/0: list -> bool;",
    ///             "ISEQUAL/0: t1. t1 -> t1 -> bool;",
    ///             "IF/0: t1. bool -> t1 -> t1 -> t1;",
    ///             "TRUE/0: bool;",
    ///             "FALSE/0: bool;",
    ///             "DIGIT/0: int -> nat;",
    ///             "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
    ///             "0/0: int; 1/0: int; 2/0: int;",
    ///             "3/0: int; 4/0: int; 5/0: int;",
    ///             "6/0: int; 7/0: int; 8/0: int;",
    ///             "9/0: int;",
    ///         ].join(" "),
    ///         &ctx
    ///     ).expect("lex");
    ///     let p_of_n_rules = |k| 0.5_f64.ln() * (k as f64) + 0.5_f64.ln();
    ///     let atom_weights = (1.0, 1.0, 1.0, 1.0);
    ///     let invent = true;
    ///
    ///     let strings = [
    ///         ".(C .(v0_ .(v1_ .(v2_ v3_)))) = .(v2_ NIL);",
    ///         // CONS after CONS DIGIT
    ///         ".(C .(v0_ .(.(CONS .(DIGIT v1_)) .(.(CONS v2_) v3_)))) = .(.(CONS v2_) NIL);",
    ///         ".(C .(.(CONS .(DIGIT v0_)) .(v1_ .(.(CONS v2_) v3_)))) = .(.(CONS v2_) NIL);",
    ///         ".(C .(.(CONS .(DIGIT v0_)) .(.(CONS v1_) .(v2_ v3_)))) = .(v2_ NIL);",
    ///         // CONS before CONS DIGIT
    ///         ".(C .(v0_ .(.(CONS v1_) .(.(CONS .(DIGIT v2_)) v3_)))) = .(.(CONS .(DIGIT v2_)) NIL);",
    ///         ".(C .(.(CONS v0_) .(v1_ .(.(CONS .(DIGIT v2_)) v3_)))) = .(.(CONS .(DIGIT v2_)) NIL);",
    ///         ".(C .(.(CONS v0_) .(.(CONS .(DIGIT v1_)) .(v2_ v3_)))) = .(v2_ NIL);",
    ///         // 3 CONS
    ///         ".(C .(.(CONS v0_) .(.(CONS v1_) .(.(CONS v2_) v3_)))) = .(.(CONS v2_) NIL);",
    ///         // 2 CONS, 1 CONS DIGIT
    ///         ".(C .(.(CONS v0_) .(.(CONS v1_) .(.(CONS .(DIGIT v2_)) v3_)))) = .(.(CONS .(DIGIT v2_)) NIL);",
    ///         ".(C .(.(CONS v0_) .(.(CONS .(DIGIT v1_)) .(.(CONS v2_) v3_)))) = .(.(CONS v2_) NIL);",
    ///         ".(C .(.(CONS .(DIGIT v0_)) .(.(CONS v1_) .(.(CONS v2_) v3_)))) = .(.(CONS v2_) NIL);",
    ///         // 1 CONS, 2 CONS DIGIT
    ///         ".(C .(.(CONS v0_) .(.(CONS .(DIGIT v1_)) .(.(CONS .(DIGIT v2_)) v3_)))) = .(.(CONS .(DIGIT v2_)) NIL);",
    ///         ".(C .(.(CONS .(DIGIT v0_)) .(.(CONS v1_) .(.(CONS .(DIGIT v2_)) v3_)))) = .(.(CONS .(DIGIT v2_)) NIL);",
    ///         ".(C .(.(CONS .(DIGIT v0_)) .(.(CONS .(DIGIT v1_)) .(.(CONS v2_) v3_)))) = .(.(CONS v2_) NIL);",
    ///         // 3 CONS DIGIT
    ///         ".(C .(.(CONS .(DIGIT v0_)) .(.(CONS .(DIGIT v1_)) .(.(CONS .(DIGIT v2_)) v3_)))) = .(.(CONS .(DIGIT v2_)) NIL);",
    ///     ];
    ///     let mut results = Vec::with_capacity(strings.len());
    ///     for string in &strings {
    ///         let trs = parse_trs(string, &mut lex, true, &[]).expect("trs");
    ///         let lp = trs.lexicon().logprior_utrs(&trs.utrs(), p_of_n_rules, atom_weights, invent).unwrap();
    ///         results.push((string, lp));
    ///     }
    ///     results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    ///     results.iter().for_each(|(s,p)| println!("{}: {}", p, s));
    /// })
    /// ```
    pub fn logprior_utrs<F>(
        &self,
        utrs: &UntypedTRS,
        p_of_n_rules: F,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError<'ctx>>
    where
        F: Fn(usize) -> f64,
    {
        let mut env = Env::new(invent, self, Some(self.lex.src));
        env.logprior_utrs(utrs, p_of_n_rules, atom_weights)
    }
    /// Give the log probability of sampling an SRS.
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::too_many_arguments))]
    pub fn logprior_srs<F>(
        &self,
        srs: &UntypedTRS,
        p_of_n_rules: F,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
    ) -> Result<f64, SampleError<'ctx>>
    where
        F: Fn(usize) -> f64,
    {
        let mut env = Env::new(invent, self, Some(self.lex.src));
        env.logprior_srs(srs, p_of_n_rules, atom_weights, dist, t_max, d_max)
    }
}

impl<'ctx> Lex<'ctx> {
    /// Add a new operator to the `Lexicon`.
    pub(crate) fn invent_operator(
        &mut self,
        name: Option<String>,
        arity: u8,
        tp: Ty<'ctx>,
    ) -> Operator {
        let op = self.sig.new_op(arity, name);
        let lex_vars = self.free_vars();
        let schema = tp.generalize(&lex_vars, &self.ctx);
        let mut free_vars = schema.free_vars();
        let type_entry = self.tps.entry(schema).or_insert_with(|| vec![]);
        type_entry.push(op);
        self.ops.push((schema, arity as usize));
        self.fvs.append(&mut free_vars);
        self.fvs.sort_unstable();
        self.fvs.dedup();
        op
    }
    /// List the free type variables in the `Lexicon`
    pub(crate) fn free_vars(&self) -> Vec<TypeVar> {
        self.fvs
            .iter()
            .flat_map(|x| self.ctx.intern_tvar(*x).vars())
            .sorted()
            .unique()
            .collect_vec()
    }
    fn recompute_free_vars(&mut self) -> &[TypeVar] {
        self.fvs.clear();
        for op in &self.ops {
            self.fvs.append(&mut op.0.free_vars())
        }
        self.fvs.sort_unstable();
        self.fvs.dedup();
        &self.fvs
    }
    fn recompute_types(&mut self) -> &HashMap<Schema<'ctx>, Vec<Operator>> {
        self.tps = HashMap::new();
        for op in self.sig.operators() {
            let entry = self
                .tps
                .entry(self.ops[op.id()].0)
                .or_insert_with(|| vec![]);
            entry.push(op);
        }
        &self.tps
    }
}

impl<'ctx> PartialEq for Lex<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        self.ops == other.ops && self.sig == other.sig
    }
}

impl<'ctx> Eq for Lex<'ctx> {}

impl<'ctx> fmt::Display for Lex<'ctx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (op, schema) in self.sig.operators().iter().zip(&self.ops) {
            writeln!(f, "{}/{}: {}", op.display(&self.sig), schema.1, schema.0,)?;
        }
        Ok(())
    }
}

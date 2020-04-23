use itertools::Itertools;
use polytype::{Context as TypeContext, Type, TypeSchema, Variable as TypeVar};
use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng,
};
use std::{
    borrow::Cow,
    collections::{HashMap, VecDeque},
    convert::TryFrom,
    f64::NEG_INFINITY,
    fmt,
};
use term_rewriting::{
    Atom, Context, MergeStrategy, Operator, PStringDist, Place, Rule, RuleContext, Signature,
    SignatureChange, Term, Variable, TRS as UntypedTRS,
};
use trs::{gp::TRSGP, Datum, SampleError, TypeError};

type SampleChoice = (Atom, Vec<Type>, Environment, TypeContext);
type RuleEnumerationPartial = (
    RuleContext,
    (usize, usize),
    Vec<Type>,
    Environment,
    TypeContext,
);

#[derive(Copy, Clone, Debug)]
pub struct SampleParams {
    atom_weights: (f64, f64, f64, f64),
    variable: bool,
    limit: GenerationLimit,
}

#[derive(Copy, Clone, Debug)]
pub enum GenerationLimit {
    TotalSize(usize),
    TermSize(usize),
}

pub struct TermEnumeration<'a> {
    stack: Vec<(Context, usize, Vec<Type>, Environment, TypeContext)>,
    limit: GenerationLimit,
    lex: &'a Lex,
}

pub struct RuleEnumeration<'a> {
    stack: Vec<RuleEnumerationPartial>,
    limit: GenerationLimit,
    lex: &'a Lex,
    invent: bool,
}

/// (representation) Manages the syntax of a term rewriting system.
#[derive(Clone, Debug)]
pub struct Lexicon<'a>(pub(crate) Cow<'a, Lex>);

#[derive(Debug)]
pub(crate) struct Lex {
    pub(crate) ops: Vec<TypeSchema>,
    pub(crate) signature: Signature,
    // We need to keep this around to make sense of self.ops, self.types, and self.free_vars.
    pub(crate) ctx: TypeContext,
    types: HashMap<TypeSchema, Vec<Operator>>,
    free_vars: Vec<TypeVar>,
}

#[derive(Clone, Debug)]
pub struct Environment {
    pub invent: bool,
    env: Vec<TypeSchema>,
    free_vars: Vec<TypeVar>,
}

fn fit_schema(
    tp: &Type,
    schema: &TypeSchema,
    constant: bool,
    ctx: &mut TypeContext,
) -> Result<Vec<Type>, SampleError> {
    let query_tp = schema.instantiate(ctx);
    let result = if constant {
        ctx.unify(&query_tp, &tp).map(|_| Vec::new())
    } else {
        ctx.unify(query_tp.returns().ok_or(TypeError::Malformed)?, tp)
            .map(|_| query_tp.args_destruct().unwrap_or_else(Vec::new))
    };
    result.map_err(SampleError::from)
}

fn compute_option_weights(
    options: &[SampleChoice],
    (vw, cw, ow, iw): (f64, f64, f64, f64),
    env: &Environment,
) -> Vec<f64> {
    options
        .iter()
        .map(|(atom, _, _, _)| match atom {
            Atom::Operator(o) if o.arity() == 0 => cw,
            Atom::Operator(_) => ow,
            Atom::Variable(v) if env.env.len() > v.id => vw,
            Atom::Variable(_) if env.invent => iw,
            _ => 0.0,
        })
        .collect_vec()
}

impl Environment {
    pub fn new(invent: bool) -> Self {
        Environment {
            invent,
            env: vec![],
            free_vars: vec![],
        }
    }
    pub fn from_context(context: &Context, types: &HashMap<Place, Type>, invent: bool) -> Self {
        // pull out the type of each variable.
        let env = context
            .subcontexts()
            .into_iter()
            .filter_map(|(t, p)| match *t {
                Context::Variable(v) => Some((v, p)),
                _ => None,
            })
            .unique_by(|(v, _)| *v)
            .sorted_by_key(|(v, _)| *v)
            .map(|(_, p)| types[&p].generalize(&[]))
            .collect_vec();
        let free_vars = env
            .iter()
            .flat_map(|schema| schema.free_vars())
            .collect_vec();
        Environment {
            env,
            invent,
            free_vars,
        }
    }
    pub fn from_rulecontext(
        rule: &RuleContext,
        types: &HashMap<Place, Type>,
        invent: bool,
    ) -> Self {
        // pull out the type of each variable.
        let env = rule
            .subcontexts()
            .into_iter()
            .filter_map(|(t, p)| match *t {
                Context::Variable(v) => Some((v, p)),
                _ => None,
            })
            .unique_by(|(v, _)| *v)
            .sorted_by_key(|(v, _)| *v)
            .map(|(_, p)| types[&p].generalize(&[]))
            .collect_vec();
        let free_vars = env
            .iter()
            .flat_map(|schema| schema.free_vars())
            .collect_vec();
        Environment {
            env,
            invent,
            free_vars,
        }
    }
    pub fn from_rule(rule: &Rule, types: &HashMap<Place, Type>, invent: bool) -> Self {
        // pull out the type of each variable.
        let env = rule
            .subterms()
            .into_iter()
            .filter_map(|(t, p)| match *t {
                Term::Variable(v) => Some((v, p)),
                _ => None,
            })
            .unique_by(|(v, _)| *v)
            .sorted_by_key(|(v, _)| *v)
            .map(|(_, p)| types[&p].generalize(&[]))
            .collect_vec();
        let free_vars = env
            .iter()
            .flat_map(|schema| schema.free_vars())
            .collect_vec();
        Environment {
            env,
            invent,
            free_vars,
        }
    }
    pub fn from_term(term: &Term, types: &HashMap<Place, Type>, invent: bool) -> Self {
        // pull out the type of each variable.
        let env = term
            .subterms()
            .into_iter()
            .filter_map(|(t, p)| match *t {
                Term::Variable(v) => Some((v, p)),
                _ => None,
            })
            .unique_by(|(v, _)| *v)
            .sorted_by_key(|(v, _)| *v)
            .map(|(_, p)| types[&p].generalize(&[]))
            .collect_vec();
        let free_vars = env
            .iter()
            .flat_map(|schema| schema.free_vars())
            .collect_vec();
        Environment {
            env,
            invent,
            free_vars,
        }
    }
    pub fn from_vars(vars: &[Variable], ctx: &mut TypeContext) -> Self {
        let env = vars
            .iter()
            .map(|_| TypeSchema::Monotype(ctx.new_variable()))
            .collect_vec();
        let mut free_vars = env
            .iter()
            .flat_map(|schema| schema.free_vars())
            .collect_vec();
        free_vars.sort_unstable();
        free_vars.dedup();
        Environment {
            invent: false,
            env,
            free_vars,
        }
    }
    pub fn len(&self) -> usize {
        self.env.len()
    }
    pub fn is_empty(&self) -> bool {
        self.env.is_empty()
    }
    pub fn invent_variable(&mut self, ctx: &mut TypeContext) -> Option<Variable> {
        if self.invent {
            let var = Variable { id: self.env.len() };
            let schema = TypeSchema::Monotype(ctx.new_variable());
            let mut free_vars = schema.free_vars();
            self.env.push(schema);
            self.free_vars.append(&mut free_vars);
            self.free_vars.sort_unstable();
            self.free_vars.dedup();
            Some(var)
        } else {
            None
        }
    }
    pub fn free_vars(&self, ctx: &mut TypeContext) -> Vec<TypeVar> {
        let mut vars = vec![];
        for x in &self.free_vars {
            let mut v = Type::Variable(*x);
            v.apply_mut_compress(ctx);
            vars.append(&mut v.vars());
        }
        vars.sort();
        vars.dedup();
        vars
    }
}

impl<'a> fmt::Display for Lexicon<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl<'a> Eq for Lexicon<'a> {}
impl<'a> PartialEq for Lexicon<'a> {
    fn eq(&self, other: &Lexicon) -> bool {
        self.0 == other.0
    }
}
impl<'a> Lexicon<'a> {
    // TODO: this is a HACK. How can we get rid of it?
    pub fn rulecontext_fillers(&self, context: &RuleContext, place: &[usize]) -> Vec<Atom> {
        if let Some(&Context::Hole) = context.at(place) {
            let mut types = HashMap::new();
            let mut ctx = self.0.ctx.clone();
            let mut env = Environment::from_vars(&context.variables(), &mut ctx);
            env.invent = true;
            self.0
                .infer_rulecontext(context, &mut types, &mut env, &mut ctx)
                .ok();
            env.invent = place[0] == 0;
            self.0
                .valid_atoms(&types[place], &env, &mut ctx)
                .into_iter()
                .filter_map(|(atom, _, _, _)| {
                    if place == [0] && atom.is_variable() {
                        None
                    } else {
                        Some(atom)
                    }
                })
                .collect()
        } else {
            vec![]
        }
    }

    /// Construct a `Lexicon` with only background [`term_rewriting::Operator`]s.
    ///
    /// # Example
    ///
    /// See [`polytype::ptp`] for details on constructing [`polytype::TypeSchema`]s.
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::Lexicon;
    /// # use polytype::Context as TypeContext;
    /// let operators = vec![
    ///     (2, Some("PLUS".to_string()), ptp![@arrow[tp!(int), tp!(int), tp!(int)]]),
    ///     (1, Some("SUCC".to_string()), ptp![@arrow[tp!(int), tp!(int)]]),
    ///     (0, Some("ZERO".to_string()), ptp![int]),
    /// ];
    ///
    /// let lexicon = Lexicon::new(operators, TypeContext::default());
    /// ```
    ///
    /// [`polytype::ptp`]: https://docs.rs/polytype/~6.0/polytype/macro.ptp.html
    /// [`polytype::TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`term_rewriting::Operator`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Operator.html
    pub fn new<'b>(
        operators: Vec<(u8, Option<String>, TypeSchema)>,
        ctx: TypeContext,
    ) -> Lexicon<'b> {
        let signature = Signature::default();
        let mut ops = Vec::with_capacity(operators.len());
        for (arity, name, tp) in operators {
            signature.new_op(arity, name);
            ops.push(tp);
        }
        let mut lex = Lexicon(Cow::Owned(Lex {
            ops,
            signature,
            ctx,
            free_vars: vec![],
            types: HashMap::new(),
        }));
        lex.0.to_mut().recompute_free_vars();
        lex.0.to_mut().recompute_types();
        lex
    }
    /// Convert a [`term_rewriting::Signature`] into a `Lexicon`:
    /// - `ops` are types for the [`term_rewriting::Operator`]s
    /// - `vars` are types for the [`term_rewriting::Variable`]s,
    /// - `background` are [`term_rewriting::Rule`]s that never change
    ///
    /// # Example
    ///
    /// See [`polytype::ptp`] for details on constructing [`polytype::TypeSchema`]s.
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::Lexicon;
    /// # use term_rewriting::Signature;
    /// # use polytype::Context as TypeContext;
    /// let mut sig = Signature::default();
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some("PLUS".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// sig.new_op(1, Some("SUCC".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    /// sig.new_op(0, Some("ZERO".to_string()));
    /// ops.push(ptp![int]);
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, TypeContext::default());
    /// ```
    ///
    /// [`polytype::ptp`]: https://docs.rs/polytype/~6.0/polytype/macro.ptp.html
    /// [`polytype::TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`term_rewriting::Signature`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Signature.html
    /// [`term_rewriting::Operator`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Operator.html
    /// [`term_rewriting::Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
    /// [`term_rewriting::Variable`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Variable.html
    pub fn from_signature<'b>(
        signature: Signature,
        ops: Vec<TypeSchema>,
        ctx: TypeContext,
    ) -> Lexicon<'b> {
        let mut lex = Lexicon(Cow::Owned(Lex {
            ops,
            signature,
            ctx,
            free_vars: vec![],
            types: HashMap::new(),
        }));
        lex.0.to_mut().recompute_free_vars();
        lex.0.to_mut().recompute_types();
        lex
    }
    /// Merge two `Lexicons` into a single `Lexicon`
    pub fn merge(
        lex1: &Lexicon<'a>,
        lex2: &Lexicon<'a>,
        strategy: MergeStrategy,
    ) -> Result<(Lexicon<'a>, SignatureChange), ()> {
        // 1. Merge the signatures.
        // TODO: expensive
        let sig = lex1.0.signature.deep_copy();
        let sig_change = sig.merge(&lex2.0.signature.deep_copy(), strategy)?;
        let mut inv_map = HashMap::new();
        for (k, v) in sig_change.op_map.iter() {
            inv_map.insert(v, k);
        }
        // 2. Merge the type contexts.
        let mut ctx = lex1.0.ctx.clone();
        // TODO: Are these the right things to share?
        let sacreds = lex2
            .0
            .free_vars
            .iter()
            .filter(|fv| lex1.0.free_vars.contains(fv))
            .cloned()
            .collect_vec();
        let ctx_change = ctx.merge(lex2.0.ctx.clone(), sacreds);
        // 3. Update ops.
        let mut ops = lex1.0.ops.clone();
        for op in &sig.operators() {
            let id = op.id();
            if id >= lex1.0.ops.len() {
                let mut schema = lex2.0.ops[*inv_map[&id]].clone();
                ctx_change.reify_typeschema(&mut schema);
                ops.push(schema);
            }
        }
        // 4. Create and return a new Lexicon from parts.
        Ok((Lexicon::from_signature(sig, ops, ctx), sig_change))
    }
    /// Return the specified `Operator` if possible.
    pub fn has_op(&self, name: Option<&str>, arity: u8) -> Result<Operator, SampleError> {
        self.0.has_op(name, arity)
    }
    /// List the free type variables in the `Lexicon`
    pub fn free_vars(&self, ctx: &mut TypeContext) -> Vec<TypeVar> {
        self.0.free_vars(ctx)
    }
    /// Add a new operator to the `Lexicon`.
    pub fn invent_operator(&mut self, name: Option<String>, arity: u8, tp: &Type) -> Operator {
        self.0.to_mut().invent_operator(name, arity, tp)
    }
    /// Return the `Lexicon`'s [`TypeContext`].
    ///
    /// [`TypeContext`]: https://docs.rs/polytype/~6.0/polytype/struct.Context.html
    pub fn context(&self) -> &TypeContext {
        &self.0.ctx
    }
    pub fn context_mut(&mut self) -> &mut TypeContext {
        &mut self.0.to_mut().ctx
    }
    /// Return the `Lexicon`'s `Signature`.
    pub fn signature(&self) -> &Signature {
        &self.0.signature
    }
    /// Enumerate `Term`s of length `n` or less.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::{parse_lexicon, Environment, Lexicon};
    /// # use polytype::{TypeSchema, Context as TypeContext};
    /// let mut lex = parse_lexicon(
    ///    &[
    ///        "A/0: term;",
    ///        "./2: term -> term -> term;",
    ///    ]
    ///        .join(" "),
    ///    TypeContext::default(),
    ///)
    ///    .expect("parsed lexicon");
    /// let mut ctx = lex.context().clone();
    /// let schema = TypeSchema::Monotype(ctx.new_variable());
    /// let env = Environment::new(false);
    /// let terms = lex.enumerate_terms(&schema, 11, &env, &ctx);
    /// assert_eq!(terms.len(), 65);
    /// ```
    pub fn enumerate_terms(
        &self,
        schema: &TypeSchema,
        n: usize,
        env: &Environment,
        ctx: &TypeContext,
    ) -> Vec<Term> {
        self.0.enumerate_terms(schema, n, env, ctx)
    }
    /// Enumerate [`Rule`]s of length `n` or less.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::{parse_lexicon, GenerationLimit, Environment, Lexicon};
    /// # use polytype::{TypeSchema, Context as TypeContext};
    /// let mut lex = parse_lexicon(
    ///    &[
    ///        "A/0: term;",
    ///        "./2: term -> term -> term;",
    ///    ]
    ///        .join(" "),
    ///    TypeContext::default(),
    ///)
    ///    .expect("parsed lexicon");
    /// let mut ctx = lex.context().clone();
    /// let schema = TypeSchema::Monotype(ctx.new_variable());
    /// let invent = false;
    /// let max_size = GenerationLimit::TotalSize(11);
    /// let terms = lex.enumerate_rules(&schema, max_size, invent, &ctx);
    /// assert_eq!(terms.len(), 64);
    /// ```
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::{parse_lexicon, GenerationLimit, Environment, Lexicon};
    /// # use polytype::{TypeSchema, Context as TypeContext};
    /// let mut lex = parse_lexicon(
    ///     &[
    ///         "C/0: list -> list;",
    ///         "CONS/0: nat -> list -> list;",
    ///         "NIL/0: list;",
    ///         "HEAD/0: list -> nat;",
    ///         "TAIL/0: list -> list;",
    ///         "EMPTY/0: list -> bool;",
    ///         "EQUAL/0: t1. t1 -> t1 -> bool;",
    ///         "IF/0: t1. bool -> t1 -> t1 -> t1;",
    ///         ">/0: nat -> nat -> bool;",
    ///         "+/0: nat -> nat -> nat;",
    ///         "-/0: nat -> nat -> nat;",
    ///         "TRUE/0: bool;",
    ///         "FALSE/0: bool;",
    ///         "DIGIT/0: int -> nat;",
    ///         "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
    ///         "0/0: int; 1/0: int; 2/0: int;",
    ///         "3/0: int; 4/0: int; 5/0: int;",
    ///         "6/0: int; 7/0: int; 8/0: int;",
    ///         "9/0: int;",
    ///     ]
    ///         .join(" "),
    ///     TypeContext::default(),
    /// )
    ///     .expect("parsed lexicon");
    /// let mut ctx = lex.context().clone();
    /// let schema = TypeSchema::Monotype(ctx.new_variable());
    /// let invent = true;
    /// let max_size = GenerationLimit::TermSize(3);
    /// let rules = lex.enumerate_rules(&schema, max_size, invent, &ctx);
    /// for rule in &rules {
    ///     println!("{}", rule.pretty(lex.signature()));
    /// }
    /// // assert_eq!(rules.len(), 121);
    /// ```
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::{parse_lexicon, GenerationLimit, Environment, Lexicon};
    /// # use polytype::{TypeSchema, Context as TypeContext};
    /// let mut lex = parse_lexicon(
    ///     &[
    ///         "ID/0: t1 . t1 -> t1;",
    ///         "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
    ///     ]
    ///         .join(" "),
    ///     TypeContext::default(),
    /// )
    ///     .expect("parsed lexicon");
    /// let mut ctx = lex.context().clone();
    /// let schema = TypeSchema::Monotype(ctx.new_variable());
    /// let invent = true;
    /// let max_size = GenerationLimit::TermSize(3);
    /// let rules = lex.enumerate_rules(&schema, max_size, invent, &ctx);
    /// for rule in &rules {
    ///     println!("{}", rule.pretty(lex.signature()));
    /// }
    /// assert_eq!(rules.len(), 18);
    /// ```
    ///
    /// [`Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
    pub fn enumerate_rules(
        &self,
        schema: &TypeSchema,
        n: GenerationLimit,
        invent: bool,
        ctx: &TypeContext,
    ) -> Vec<Rule> {
        self.0.enumerate_rules(schema, n, invent, ctx)
    }
    /// Infer the [`TypeSchema`] associated with an [`Operator`].
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`Operator`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Variable.html
    pub fn infer_operator(&self, op: Operator) -> Result<TypeSchema, TypeError> {
        self.0.op_tp(op).map(|tp| tp.clone())
    }
    /// Infer the [`TypeSchema`] associated with a [`Variable`].
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`Variable`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Variable.html
    pub fn infer_variable(
        &self,
        var: Variable,
        env: &Environment,
    ) -> Result<TypeSchema, TypeError> {
        self.0.var_tp(var, env).map(|tp| tp.clone())
    }
    /// Infer the [`TypeSchema`] associated with an [`Atom`].
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`Atom`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Atom.html
    pub fn infer_atom(&self, atom: &Atom, env: &Environment) -> Result<TypeSchema, TypeError> {
        self.0.infer_atom(atom, env).map(|tp| tp.clone())
    }
    /// Infer the [`TypeSchema`] associated with a [`Term`].
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`Term`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Term.html
    pub fn infer_term(
        &self,
        term: &Term,
        types: &mut HashMap<Place, Type>,
        env: &mut Environment,
        ctx: &mut TypeContext,
    ) -> Result<TypeSchema, TypeError> {
        self.0.infer_term(term, types, env, ctx)
    }
    /// Infer the [`TypeSchema`] associated with a [`Rule`].
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
    pub fn infer_rule(
        &self,
        rule: &Rule,
        types: &mut HashMap<Place, Type>,
        env: &mut Environment,
        ctx: &mut TypeContext,
    ) -> Result<TypeSchema, TypeError> {
        self.0.infer_rule(rule, types, env, ctx)
    }
    /// Infer the [`TypeSchema`] associated with a slice of [`Rule`]s.
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
    pub fn infer_data(
        &self,
        data: &[Datum],
        ctx: &mut TypeContext,
    ) -> Result<TypeSchema, TypeError> {
        self.0.infer_data(data, ctx)
    }
    /// Infer the [`TypeSchema`] associated with a [`Context`].
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{parse_lexicon, parse_context, Environment, Lexicon};
    /// # use polytype::Context as TypeContext;
    /// # use std::collections::HashMap;
    /// let mut lex = parse_lexicon(
    ///     "PLUS/2: int -> int -> int; SUCC/1: int -> int; ZERO/0: int;",
    ///     TypeContext::default())
    /// .expect("parsed lexicon");
    /// let context = parse_context("SUCC([!])", &mut lex).expect("parsed context");
    /// let mut ctx = lex.context().clone();
    /// let mut env = Environment::from_vars(&context.variables(), &mut ctx);
    /// let mut types = HashMap::new();
    /// let inferred_schema = lex.infer_context(&context, &mut types, &mut env, &mut ctx).unwrap();
    ///
    /// assert_eq!("int", inferred_schema.to_string());
    /// ```
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`Context`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Context.html
    pub fn infer_context(
        &self,
        context: &Context,
        types: &mut HashMap<Place, Type>,
        env: &mut Environment,
        ctx: &mut TypeContext,
    ) -> Result<TypeSchema, TypeError> {
        self.0.infer_context(context, types, env, ctx)
    }
    /// Infer the [`TypeSchema`] associated with a [`RuleContext`].
    ///
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`RuleContext`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.RuleContext.html
    pub fn infer_rulecontext(
        &self,
        context: &RuleContext,
        types: &mut HashMap<Place, Type>,
        env: &mut Environment,
        ctx: &mut TypeContext,
    ) -> Result<TypeSchema, TypeError> {
        self.0.infer_rulecontext(context, types, env, ctx)
    }
    /// Infer the [`TypeSchema`] associated with a [`TRS`].
    /// [`TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`RuleContext`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.RuleContext.html
    pub fn infer_utrs(&self, utrs: &UntypedTRS, ctx: &mut TypeContext) -> Result<(), TypeError> {
        self.0.infer_utrs(utrs, ctx)
    }
    /// Sample a [`term_rewriting::Term`].
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # use programinduction::trs::{Environment, parse_lexicon, Lexicon};
    /// # use polytype::Context as TypeContext;
    /// # use rand::{thread_rng};
    /// let mut lex = parse_lexicon(
    ///     "PLUS/2: int -> int -> int; SUCC/1: int -> int; ZERO/0: int;",
    ///     TypeContext::default())
    ///     .expect("parsed lexicon");
    /// let schema = ptp![int];
    /// let atom_weights = (2.0, 2.0, 1.0, 2.0);
    /// let variable = true;
    /// let max_size = 20;
    /// let env = Environment::new(true);
    /// let mut ctx = lex.context().clone();
    /// let mut rng = thread_rng();
    ///
    /// for i in 0..50 {
    ///     let term = lex.sample_term(&schema, atom_weights, variable, max_size, &env, &mut ctx, &mut rng).unwrap();
    ///     println!("{}. {}", i, term.pretty(&lex.signature()));
    /// }
    /// ```
    ///
    /// [`term_rewriting::Term`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Term.html
    #[allow(clippy::too_many_arguments)]
    pub fn sample_term<R: Rng>(
        &self,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        variable: bool,
        max_size: usize,
        env: &Environment,
        ctx: &mut TypeContext,
        rng: &mut R,
    ) -> Result<Term, SampleError> {
        self.0
            .sample_term(schema, atom_weights, variable, max_size, env, ctx, rng)
    }
    /// Sample a `Term` conditioned on a `Context` rather than a `TypeSchema`.
    pub fn sample_term_from_context<R: Rng>(
        &self,
        context: &Context,
        arg_types: &mut Vec<Type>,
        params: SampleParams,
        env: &Environment,
        ctx: &mut TypeContext,
        rng: &mut R,
    ) -> Result<Term, SampleError> {
        self.0
            .sample_term_internal(context, arg_types, params, env, ctx, rng)
    }
    /// Sample a `Rule`.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # use programinduction::trs::{GenerationLimit, parse_lexicon, Lexicon};
    /// # use polytype::Context as TypeContext;
    /// # use rand::{thread_rng};
    /// let mut lex = parse_lexicon(
    ///     "PLUS/2: int -> int -> int; SUCC/1: int -> int; ZERO/0: int;",
    ///     TypeContext::default())
    ///     .expect("parsed lexicon");
    /// let schema = ptp![int];
    /// let atom_weights = (3.0, 3.0, 1.0, 3.0);
    /// let limit = GenerationLimit::TotalSize(40);
    /// let invent = true;
    /// let mut ctx = lex.context().clone();
    /// let mut rng = thread_rng();
    ///
    /// for i in 0..50 {
    ///     let rule = lex.sample_rule(&schema, atom_weights, limit, invent, &mut ctx, &mut rng).unwrap();
    ///     println!("{}. {}", i, rule.pretty(&lex.signature()));
    /// }
    /// ```
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # use programinduction::trs::{GenerationLimit, parse_lexicon, Lexicon};
    /// # use polytype::Context as TypeContext;
    /// # use rand::{thread_rng};
    /// let mut lex = parse_lexicon(
    ///     "equal/0: t1. t1 -> t1 -> bool; >/0: int -> int -> bool; ./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
    ///     TypeContext::default())
    ///     .expect("parsed lexicon");
    /// let schema = ptp![0];
    /// let atom_weights = (3.0, 3.0, 1.0, 3.0);
    /// let limit = GenerationLimit::TermSize(1);
    /// let invent = true;
    /// let mut ctx = lex.context().clone();
    /// let mut rng = thread_rng();
    ///
    /// let mut rules = vec![];
    /// while rules.len() < 500 {
    ///     if let Ok(rule) = lex.sample_rule(&schema, atom_weights, limit, invent, &mut ctx.clone(), &mut rng) {
    ///         rules.push(rule.pretty(&lex.signature()));
    ///         println!("{}", rule.pretty(&lex.signature()));
    ///     }
    /// }
    /// assert!(rules.contains(&"> = equal".to_string()));
    /// assert!(!rules.contains(&"equal = >".to_string()));
    /// ```
    pub fn sample_rule<R: Rng>(
        &self,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        limit: GenerationLimit,
        invent: bool,
        ctx: &mut TypeContext,
        rng: &mut R,
    ) -> Result<Rule, SampleError> {
        self.0
            .sample_rule(schema, atom_weights, limit, invent, ctx, rng)
    }
    /// Sample a `Rule` conditioned on a `Context` rather than a `TypeSchema`.
    pub fn sample_rule_from_context<R: Rng>(
        &self,
        context: &RuleContext,
        atom_weights: (f64, f64, f64, f64),
        limit: GenerationLimit,
        invent: bool,
        ctx: &mut TypeContext,
        rng: &mut R,
    ) -> Result<Rule, SampleError> {
        self.0
            .sample_rule_from_context(context, atom_weights, limit, invent, ctx, rng)
    }
    /// Give the log probability of sampling a `Term`.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::{parse_lexicon, parse_term, Environment, Lexicon};
    /// # use polytype::{TypeSchema, Context as TypeContext};
    /// let mut lex = parse_lexicon(
    ///    &[
    ///        "A/0: term;",
    ///        "./2: term -> term -> term;",
    ///    ]
    ///        .join(" "),
    ///    TypeContext::default(),
    ///)
    ///    .expect("parsed lexicon");
    /// let term = parse_term("(A ((v0_ v1_) (v1_ (A v0_))))", &mut lex).expect("parsed term");
    /// let mut ctx = lex.context().clone();
    /// let schema = TypeSchema::Monotype(ctx.new_variable());
    /// let atom_weights = (1.0, 1.0, 1.0, 1.0);
    /// let mut env = Environment::new(true);
    /// let lp = lex.logprior_term(&term, &schema, atom_weights, &mut env, &mut ctx).unwrap();
    /// assert_eq!((lp*1e9_f64).round(), ((-5.0*3.0_f64.ln() - 4.0_f64.ln() - 5.0*5.0_f64.ln())*1e9_f64).round());
    /// ```
    pub fn logprior_term(
        &self,
        term: &Term,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        env: &mut Environment,
        ctx: &mut TypeContext,
    ) -> Result<f64, SampleError> {
        self.0.logprior_term(term, schema, atom_weights, env, ctx)
    }
    /// Give the log probability of sampling a `Rule`.
    pub fn logprior_rule(
        &self,
        rule: &Rule,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        ctx: &mut TypeContext,
    ) -> Result<f64, SampleError> {
        self.0.logprior_rule(rule, atom_weights, invent, ctx)
    }
    /// Give the log probability of sampling a TRS.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::{parse_lexicon, parse_trs, Environment, Lexicon};
    /// # use polytype::{TypeSchema, Context as TypeContext};
    /// let mut lex = parse_lexicon(
    ///     &[
    ///         "C/0: list -> list;",
    ///         "CONS/0: nat -> list -> list;",
    ///         "NIL/0: list;",
    ///         "HEAD/0: list -> nat;",
    ///         "TAIL/0: list -> list;",
    ///         "ISEMPTY/0: list -> bool;",
    ///         "ISEQUAL/0: t1. t1 -> t1 -> bool;",
    ///         "IF/0: t1. bool -> t1 -> t1 -> t1;",
    ///         "TRUE/0: bool;",
    ///         "FALSE/0: bool;",
    ///         "DIGIT/0: int -> nat;",
    ///         "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
    ///         "0/0: int; 1/0: int; 2/0: int;",
    ///         "3/0: int; 4/0: int; 5/0: int;",
    ///         "6/0: int; 7/0: int; 8/0: int;",
    ///         "9/0: int;",
    ///     ]
    ///         .join(" "),
    ///     TypeContext::default(),
    /// )
    ///     .expect("parsed lexicon");
    /// let p_of_n_rules = |k| 0.5_f64.ln() * (k as f64) + 0.5_f64.ln();
    /// let atom_weights = (1.0, 1.0, 1.0, 1.0);
    /// let invent = true;
    ///
    /// let strings = [
    ///     ".(C .(v0_ .(v1_ .(v2_ v3_)))) = .(v2_ NIL);",
    ///     // CONS after CONS DIGIT
    ///     ".(C .(v0_ .(.(CONS .(DIGIT v1_)) .(.(CONS v2_) v3_)))) = .(.(CONS v2_) NIL);",
    ///     ".(C .(.(CONS .(DIGIT v0_)) .(v1_ .(.(CONS v2_) v3_)))) = .(.(CONS v2_) NIL);",
    ///     ".(C .(.(CONS .(DIGIT v0_)) .(.(CONS v1_) .(v2_ v3_)))) = .(v2_ NIL);",
    ///     // CONS before CONS DIGIT
    ///     ".(C .(v0_ .(.(CONS v1_) .(.(CONS .(DIGIT v2_)) v3_)))) = .(.(CONS .(DIGIT v2_)) NIL);",
    ///     ".(C .(.(CONS v0_) .(v1_ .(.(CONS .(DIGIT v2_)) v3_)))) = .(.(CONS .(DIGIT v2_)) NIL);",
    ///     ".(C .(.(CONS v0_) .(.(CONS .(DIGIT v1_)) .(v2_ v3_)))) = .(v2_ NIL);",
    ///     // 3 CONS
    ///     ".(C .(.(CONS v0_) .(.(CONS v1_) .(.(CONS v2_) v3_)))) = .(.(CONS v2_) NIL);",
    ///     // 2 CONS, 1 CONS DIGIT
    ///     ".(C .(.(CONS v0_) .(.(CONS v1_) .(.(CONS .(DIGIT v2_)) v3_)))) = .(.(CONS .(DIGIT v2_)) NIL);",
    ///     ".(C .(.(CONS v0_) .(.(CONS .(DIGIT v1_)) .(.(CONS v2_) v3_)))) = .(.(CONS v2_) NIL);",
    ///     ".(C .(.(CONS .(DIGIT v0_)) .(.(CONS v1_) .(.(CONS v2_) v3_)))) = .(.(CONS v2_) NIL);",
    ///     // 1 CONS, 2 CONS DIGIT
    ///     ".(C .(.(CONS v0_) .(.(CONS .(DIGIT v1_)) .(.(CONS .(DIGIT v2_)) v3_)))) = .(.(CONS .(DIGIT v2_)) NIL);",
    ///     ".(C .(.(CONS .(DIGIT v0_)) .(.(CONS v1_) .(.(CONS .(DIGIT v2_)) v3_)))) = .(.(CONS .(DIGIT v2_)) NIL);",
    ///     ".(C .(.(CONS .(DIGIT v0_)) .(.(CONS .(DIGIT v1_)) .(.(CONS v2_) v3_)))) = .(.(CONS v2_) NIL);",
    ///     // 3 CONS DIGIT
    ///     ".(C .(.(CONS .(DIGIT v0_)) .(.(CONS .(DIGIT v1_)) .(.(CONS .(DIGIT v2_)) v3_)))) = .(.(CONS .(DIGIT v2_)) NIL);",
    /// ];
    /// let mut results = Vec::with_capacity(strings.len());
    /// for string in &strings {
    ///     let trs = parse_trs(string, &mut lex, true, &[]).expect("parsed trs");
    ///     let mut ctx = lex.context().clone();
    ///     let lp = lex.logprior_utrs(&trs.utrs(), p_of_n_rules, atom_weights, invent, &mut ctx).unwrap();
    ///     results.push((string, lp));
    /// }
    /// results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    /// results.iter().for_each(|(s,p)| println!("{}: {}", p, s));
    /// ```
    pub fn logprior_utrs<F>(
        &self,
        utrs: &UntypedTRS,
        p_of_n_rules: F,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        ctx: &mut TypeContext,
    ) -> Result<f64, SampleError>
    where
        F: Fn(usize) -> f64,
    {
        self.0
            .logprior_utrs(utrs, p_of_n_rules, atom_weights, invent, ctx)
    }
    /// Give the log probability of sampling an SRS.
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::too_many_arguments))]
    pub fn logprior_srs<F>(
        &self,
        utrs: &UntypedTRS,
        p_of_n_rules: F,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
        ctx: &mut TypeContext,
    ) -> Result<f64, SampleError>
    where
        F: Fn(usize) -> f64,
    {
        self.0.logprior_srs(
            utrs,
            p_of_n_rules,
            atom_weights,
            invent,
            dist,
            t_max,
            d_max,
            ctx,
        )
    }
}

impl PartialEq for Lex {
    fn eq(&self, other: &Self) -> bool {
        self.ops == other.ops && self.signature == other.signature
    }
}
impl Eq for Lex {}
impl std::clone::Clone for Lex {
    fn clone(&self) -> Self {
        Lex {
            ops: self.ops.clone(),
            free_vars: self.free_vars.clone(),
            types: self.types.clone(),
            // TODO: confirm that we don't need a deep copy here.
            signature: self.signature.deep_copy(),
            ctx: self.ctx.clone(),
        }
    }
}
impl fmt::Display for Lex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (op, schema) in self.signature.operators().iter().zip(&self.ops) {
            writeln!(
                f,
                "{}/{}: {}",
                op.display(&self.signature),
                op.arity(),
                schema
            )?;
        }
        Ok(())
    }
}
impl Lex {
    fn recompute_free_vars(&mut self) -> &[TypeVar] {
        self.free_vars.clear();
        for op in &self.ops {
            self.free_vars.append(&mut op.free_vars())
        }
        self.free_vars.sort_unstable();
        self.free_vars.dedup();
        &self.free_vars
    }
    fn recompute_types(&mut self) -> &HashMap<TypeSchema, Vec<Operator>> {
        self.types = HashMap::new();
        for op in self.signature.operators() {
            let entry = self
                .types
                .entry(self.ops[op.id()].clone())
                .or_insert_with(|| vec![]);
            entry.push(op);
        }
        &self.types
    }
    fn has_op(&self, name: Option<&str>, arity: u8) -> Result<Operator, SampleError> {
        self.signature
            .has_op(arity, name.map(String::from))
            .ok_or(SampleError::OptionsExhausted)
    }
    fn free_vars(&self, ctx: &mut TypeContext) -> Vec<TypeVar> {
        let mut vars = vec![];
        for x in &self.free_vars {
            let mut v = Type::Variable(*x);
            v.apply_mut_compress(ctx);
            vars.append(&mut v.vars());
        }
        vars.sort();
        vars.dedup();
        vars
    }
    fn invent_operator(&mut self, name: Option<String>, arity: u8, tp: &Type) -> Operator {
        let op = self.signature.new_op(arity, name);
        let lex_vars = self.free_vars(&mut self.ctx.clone());
        let schema = tp.generalize(&lex_vars);
        let mut free_vars = schema.free_vars();
        let type_entry = self.types.entry(schema.clone()).or_insert_with(|| vec![]);
        type_entry.push(op);
        self.ops.push(schema);
        self.free_vars.append(&mut free_vars);
        self.free_vars.sort_unstable();
        self.free_vars.dedup();
        op
    }
    fn enumerate_terms(
        &self,
        schema: &TypeSchema,
        n: usize,
        env: &Environment,
        ctx: &TypeContext,
    ) -> Vec<Term> {
        TermEnumeration::new(self, schema, n, env, ctx).collect_vec()
    }
    fn enumerate_rules(
        &self,
        schema: &TypeSchema,
        n: GenerationLimit,
        invent: bool,
        ctx: &TypeContext,
    ) -> Vec<Rule> {
        RuleEnumeration::new(self, schema, n, invent, ctx).collect_vec()
    }
    fn op_tp(&self, o: Operator) -> Result<&TypeSchema, TypeError> {
        let o_id = o.id();
        if o_id >= self.ops.len() {
            Err(TypeError::NotFound)
        } else {
            Ok(&self.ops[o_id])
        }
    }
    fn var_tp<'a>(
        &'a self,
        v: Variable,
        env: &'a Environment,
    ) -> Result<&'a TypeSchema, TypeError> {
        let v_id = v.id();
        if v_id >= env.env.len() {
            Err(TypeError::NotFound)
        } else {
            Ok(&env.env[v_id])
        }
    }
    fn infer_atom<'a>(
        &'a self,
        atom: &Atom,
        env: &'a Environment,
    ) -> Result<&'a TypeSchema, TypeError> {
        match *atom {
            Atom::Operator(o) => self.op_tp(o),
            Atom::Variable(v) => self.var_tp(v, env),
        }
    }
    fn infer_term(
        &self,
        term: &Term,
        types: &mut HashMap<Place, Type>,
        env: &mut Environment,
        ctx: &mut TypeContext,
    ) -> Result<TypeSchema, TypeError> {
        let mut place = vec![];
        self.infer_term_internal(term, &mut place, types, &env, ctx)?;
        for (_, v) in types.iter_mut() {
            v.apply_mut_compress(ctx);
        }
        let mut lex_vars = self.free_vars(ctx);
        lex_vars.append(&mut env.free_vars(ctx));
        Ok(types[&place].apply_compress(ctx).generalize(&lex_vars))
    }
    fn infer_term_internal(
        &self,
        term: &Term,
        place: &mut Place,
        tps: &mut HashMap<Place, Type>,
        env: &Environment,
        ctx: &mut TypeContext,
    ) -> Result<Type, TypeError> {
        let tp = match *term {
            Term::Variable(v) => self.var_tp(v, env)?.instantiate(ctx),
            Term::Application { op, .. } if op.arity() == 0 => {
                self.op_tp(op)?.instantiate(ctx).apply_compress(ctx)
            }
            Term::Application { op, ref args } => {
                let head_type = self.op_tp(op)?.instantiate(ctx);
                let arg_types = head_type.args().unwrap();
                for (i, (a, tp)) in args.iter().zip(arg_types).enumerate() {
                    place.push(i);
                    self.infer_term_internal(a, place, tps, env, ctx)?;
                    ctx.unify(tp, &tps[place])?;
                    place.pop();
                }
                let return_tp = head_type.returns().unwrap_or(&head_type);
                return_tp.apply_compress(ctx)
            }
        };
        tps.insert(place.to_vec(), tp.clone());
        Ok(tp)
    }
    fn infer_rule(
        &self,
        rule: &Rule,
        types: &mut HashMap<Place, Type>,
        env: &mut Environment,
        ctx: &mut TypeContext,
    ) -> Result<TypeSchema, TypeError> {
        let lhs_type = self.infer_term_internal(&rule.lhs, &mut vec![0], types, &env, ctx)?;
        let var = ctx.new_variable().vars().pop().unwrap();
        ctx.extend(var, lhs_type.clone());
        let rhs_types = rule
            .rhs
            .iter()
            .enumerate()
            .map(|(i, rhs)| {
                self.infer_term_internal(&rhs, &mut vec![i + 1], types, &env, ctx)
                    .map(|_| types[&vec![i + 1]].clone())
            })
            .collect::<Result<Vec<Type>, _>>()?;
        // match LHS against RHS (order matters; RHS must be at least as general as LHS)
        for rhs_type in rhs_types {
            ctx.unify(&rhs_type, &lhs_type)?;
        }
        for (_, v) in types.iter_mut() {
            v.apply_mut_compress(ctx);
        }
        let mut lex_vars = self.free_vars(ctx);
        lex_vars.append(&mut env.free_vars(ctx));
        Ok(lhs_type.apply_compress(ctx).generalize(&lex_vars))
    }
    fn infer_data(&self, data: &[Datum], ctx: &mut TypeContext) -> Result<TypeSchema, TypeError> {
        let data_tps = data
            .iter()
            .map(|datum| match datum {
                Datum::Full(rule) => {
                    let mut env = Environment::from_vars(&rule.variables(), ctx);
                    self.infer_rule(rule, &mut HashMap::new(), &mut env, ctx)
                        .map(|tp| tp.instantiate(ctx))
                }
                Datum::Partial(term) => {
                    let mut env = Environment::from_vars(&term.variables(), ctx);
                    self.infer_term(term, &mut HashMap::new(), &mut env, ctx)
                        .map(|tp| tp.instantiate(ctx))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tp = ctx.new_variable();
        for rule_tp in data_tps {
            ctx.unify(&tp, &rule_tp)?;
        }
        let lex_vars = self.free_vars(ctx);
        Ok(tp.apply_compress(ctx).generalize(&lex_vars))
    }
    fn infer_context(
        &self,
        context: &Context,
        types: &mut HashMap<Place, Type>,
        env: &mut Environment,
        ctx: &mut TypeContext,
    ) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_context_internal(context, &mut vec![], types, &env, ctx)?;
        for (_, v) in types.iter_mut() {
            v.apply_mut_compress(ctx);
        }
        let lex_vars = self.free_vars(ctx);
        Ok(tp.apply_compress(ctx).generalize(&lex_vars))
    }
    fn infer_context_internal(
        &self,
        context: &Context,
        place: &mut Place,
        tps: &mut HashMap<Place, Type>,
        env: &Environment,
        ctx: &mut TypeContext,
    ) -> Result<Type, TypeError> {
        let tp = match *context {
            Context::Hole => ctx.new_variable(),
            Context::Variable(v) => self.var_tp(v, env)?.instantiate(ctx),
            Context::Application { op, ref args } => {
                let head_type = self.op_tp(op)?.instantiate(ctx);
                let head_args = head_type
                    .args()
                    .unwrap_or_else(|| VecDeque::with_capacity(0));
                // Applicative systems may have args.len() < head_args.len().
                if head_args.len() < args.len() {
                    return Err(TypeError::Malformed);
                }
                for (i, (arg, tp)) in args.iter().zip(head_args).enumerate() {
                    place.push(i);
                    self.infer_context_internal(arg, place, tps, &env, ctx)?;
                    ctx.unify(tp, &tps[place])?;
                    place.pop();
                }
                let return_tp = if op.arity() > 0 {
                    head_type.returns().unwrap_or(&head_type)
                } else {
                    &head_type
                };
                return_tp.apply_compress(ctx)
            }
        };
        tps.insert(place.clone(), tp.clone());
        Ok(tp)
    }
    fn infer_rulecontext(
        &self,
        context: &RuleContext,
        types: &mut HashMap<Place, Type>,
        env: &mut Environment,
        ctx: &mut TypeContext,
    ) -> Result<TypeSchema, TypeError> {
        let lhs_type = self.infer_context_internal(&context.lhs, &mut vec![0], types, &env, ctx)?;
        let var = ctx.new_variable().vars().pop().unwrap();
        ctx.extend(var, lhs_type.clone());
        let rhs_types = context
            .rhs
            .iter()
            .enumerate()
            .map(|(i, rhs)| self.infer_context_internal(&rhs, &mut vec![i + 1], types, &env, ctx))
            .collect::<Result<Vec<Type>, _>>()?;
        // match LHS against RHS (order matters; RHS must be at least as general as LHS)
        for rhs_type in rhs_types {
            ctx.unify(&rhs_type, &lhs_type)?;
        }
        for (_, v) in types.iter_mut() {
            v.apply_mut_compress(ctx);
        }
        let mut lex_vars = self.free_vars(ctx);
        lex_vars.append(&mut env.free_vars(ctx));
        Ok(lhs_type.apply_compress(ctx).generalize(&lex_vars))
    }
    fn infer_utrs(&self, utrs: &UntypedTRS, ctx: &mut TypeContext) -> Result<(), TypeError> {
        for rule in &utrs.rules {
            let mut env = Environment::from_vars(&rule.variables(), ctx);
            self.infer_rule(rule, &mut HashMap::new(), &mut env, ctx)?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn sample_atom<R: Rng>(
        &self,
        atom_weights: (f64, f64, f64, f64),
        variable: bool,
        tp: &Type,
        env: &Environment,
        ctx: &mut TypeContext,
        rng: &mut R,
    ) -> Result<SampleChoice, SampleError> {
        // List all the options.
        let mut options = self
            .valid_atoms(tp, env, ctx)
            .into_iter()
            .filter(|(atom, _, _, _)| match atom {
                Atom::Variable(_) => variable,
                _ => true,
            })
            .collect_vec();
        if options.is_empty() {
            return Err(SampleError::OptionsExhausted);
        }
        // Weight the options.
        let weights = compute_option_weights(&options, atom_weights, env);
        // Sample an option.
        let indices = WeightedIndex::new(weights).unwrap();
        Ok(options.swap_remove(indices.sample(rng)))
    }
    #[allow(clippy::too_many_arguments)]
    fn sample_term<R: Rng>(
        &self,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        variable: bool,
        max_size: usize,
        env: &Environment,
        ctx: &mut TypeContext,
        rng: &mut R,
    ) -> Result<Term, SampleError> {
        let params = SampleParams {
            variable,
            atom_weights,
            limit: GenerationLimit::TotalSize(max_size),
        };
        let partial = Context::Hole;
        let mut tp = schema.instantiate(ctx);
        tp.apply_mut_compress(ctx);
        let mut arg_types = vec![tp];
        self.sample_term_internal(&partial, &mut arg_types, params, env, ctx, rng)
    }
    fn sample_term_internal<R: Rng>(
        &self,
        context: &Context,
        arg_types: &mut Vec<Type>,
        params: SampleParams,
        env: &Environment,
        ctx: &mut TypeContext,
        rng: &mut R,
    ) -> Result<Term, SampleError> {
        let mut size = [context.size()];
        if !params.limit.is_okay(&size) {
            return Err(SampleError::SizeExceeded);
        }
        let mut env = env.clone();
        let mut partial = context.clone();
        loop {
            match Term::try_from(&partial) {
                Ok(term) => return Ok(term),
                Err(hole_place) => {
                    let tp = arg_types[0].apply_compress(ctx);
                    let variable = params.variable;
                    let (atom, mut new_arg_types, new_env, new_ctx) =
                        self.sample_atom(params.atom_weights, variable, &tp, &env, ctx, rng)?;
                    let subcontext = Context::from(atom);
                    size[0] += subcontext.size() - 1;
                    if params.limit.is_okay(&size) {
                        partial = partial.replace(&hole_place, subcontext).unwrap();
                        new_arg_types.extend_from_slice(&arg_types[1..]);
                        *arg_types = new_arg_types;
                        *ctx = new_ctx;
                        env = new_env;
                    } else {
                        return Err(SampleError::SizeExceeded);
                    }
                }
            }
        }
    }
    fn sample_rule<R: Rng>(
        &self,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        limit: GenerationLimit,
        invent: bool,
        ctx: &mut TypeContext,
        rng: &mut R,
    ) -> Result<Rule, SampleError> {
        let params = SampleParams {
            variable: true,
            atom_weights,
            limit,
        };
        let partial = RuleContext::default();
        let mut tp = schema.instantiate(ctx);
        tp.apply_mut_compress(ctx);
        let mut arg_types = vec![tp.clone(), tp];
        let env = Environment::new(invent);
        self.sample_rule_internal(&partial, &mut arg_types, params, &env, ctx, rng)
    }
    fn sample_rule_from_context<R: Rng>(
        &self,
        partial: &RuleContext,
        atom_weights: (f64, f64, f64, f64),
        limit: GenerationLimit,
        invent: bool,
        ctx: &mut TypeContext,
        rng: &mut R,
    ) -> Result<Rule, SampleError> {
        let params = SampleParams {
            variable: true,
            atom_weights,
            limit,
        };
        let mut types = HashMap::new();
        let mut env = Environment::from_vars(&partial.variables(), ctx);
        env.invent = invent;
        self.infer_rulecontext(&partial, &mut types, &mut env, ctx)?;
        let mut arg_types = partial
            .holes()
            .iter()
            .map(|h| types[h].apply_compress(ctx))
            .collect_vec();
        self.sample_rule_internal(&partial, &mut arg_types, params, &env, ctx, rng)
    }
    fn sample_rule_internal<R: Rng>(
        &self,
        context: &RuleContext,
        arg_types: &mut Vec<Type>,
        params: SampleParams,
        env: &Environment,
        ctx: &mut TypeContext,
        rng: &mut R,
    ) -> Result<Rule, SampleError> {
        let mut size = std::iter::once(context.lhs.size())
            .chain(context.rhs.iter().map(|rhs| rhs.size()))
            .collect_vec();
        if !params.limit.is_okay(&size) {
            return Err(SampleError::SizeExceeded);
        }
        let mut env = env.clone();
        let invent = env.invent;
        let mut partial = context.clone();
        loop {
            match Rule::try_from(&partial) {
                Ok(rule) => return Ok(rule),
                Err(hole_place) => {
                    let lhs_hole = hole_place[0] == 0;
                    env.invent = invent && lhs_hole && hole_place != [0];
                    let tp = arg_types[0].apply_compress(ctx);
                    let variable = hole_place != [0] && params.variable;
                    let (atom, mut new_arg_types, new_env, new_ctx) =
                        self.sample_atom(params.atom_weights, variable, &tp, &env, ctx, rng)?;
                    let subcontext = Context::from(atom);
                    size[hole_place[0]] += subcontext.size() - 1;
                    if params.limit.is_okay(&size) {
                        partial = partial.replace(&hole_place, subcontext).unwrap();
                        new_arg_types.extend_from_slice(&arg_types[1..]);
                        *arg_types = new_arg_types;
                        *ctx = new_ctx;
                        env = new_env;
                    } else {
                        return Err(SampleError::SizeExceeded);
                    }
                }
            }
        }
    }
    fn valid_atoms(
        &self,
        tp: &Type,
        env: &Environment,
        ctx: &mut TypeContext,
    ) -> Vec<SampleChoice> {
        let tp = tp.apply_compress(ctx);
        let mut atoms = Vec::with_capacity(self.ops.len() + env.env.len() + (env.invent as usize));
        for (schema, ops) in self.types.iter() {
            let mut results = [None, None];
            let mut new_ctxs = [ctx.clone(), ctx.clone()];
            for &op in ops {
                let class = (op.arity() != 0) as usize;
                if results[class].is_none() {
                    let fit = fit_schema(&tp, schema, class == 0, &mut new_ctxs[class]);
                    results[class] = Some(fit);
                }
                if let Some(Ok(ref arg_types)) = results[class] {
                    let new_ctx = new_ctxs[class].clone();
                    atoms.push((Atom::Operator(op), arg_types.clone(), env.clone(), new_ctx))
                }
            }
        }
        for (id, schema) in env.env.iter().enumerate() {
            let mut new_ctx = ctx.clone();
            if let Ok(arg_types) = fit_schema(&tp, schema, true, &mut new_ctx) {
                let atom = Atom::Variable(Variable { id });
                atoms.push((atom, arg_types, env.clone(), new_ctx));
            }
        }
        let mut new_ctx = ctx.clone();
        let mut new_env = env.clone();
        if let Some(new_var) = new_env.invent_variable(&mut new_ctx) {
            let schema = self.var_tp(new_var, &new_env).unwrap();
            let arg_types = fit_schema(&tp, schema, true, &mut new_ctx).unwrap();
            atoms.push((Atom::Variable(new_var), arg_types, new_env, new_ctx));
        }
        atoms
    }
    fn compute_prior_weight(
        &self,
        atom_weights: (f64, f64, f64, f64),
        tp: &Type,
        env: &Environment,
        ctx: &mut TypeContext,
        head: Atom,
    ) -> Option<(SampleChoice, f64, f64)> {
        let mut options = self.valid_atoms(tp, env, ctx);
        let mut weights = compute_option_weights(&options, atom_weights, env);
        let z: f64 = weights.iter().sum();
        let idx = options.iter().position(|(atom, _, _, _)| *atom == head);
        idx.map(|idx| (options.swap_remove(idx), weights.swap_remove(idx), z))
    }
    fn logprior_term(
        &self,
        term: &Term,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        env: &mut Environment,
        ctx: &mut TypeContext,
    ) -> Result<f64, SampleError> {
        self.logprior_term_internal(term, &schema.instantiate(ctx), atom_weights, env, ctx)
    }
    fn logprior_term_internal(
        &self,
        term: &Term,
        tp: &Type,
        atom_weights: (f64, f64, f64, f64),
        env: &mut Environment,
        ctx: &mut TypeContext,
    ) -> Result<f64, SampleError> {
        let mut stack = vec![(term.clone(), tp.clone())];
        let mut lp = 0.0;
        while let Some((term, mut tp)) = stack.pop() {
            // Update the type.
            tp.apply_mut_compress(ctx);
            // Collect all options.
            let option = self.compute_prior_weight(atom_weights, &tp, env, ctx, term.head());
            // Compute the probability of the term.
            match option {
                None => {
                    return Ok(NEG_INFINITY);
                }
                Some(((Atom::Variable(_), _, new_env, new_ctx), w, z)) => {
                    *env = new_env;
                    *ctx = new_ctx;
                    lp += w.ln() - z.ln();
                }
                Some(((Atom::Operator(_), arg_types, new_env, new_ctx), w, z)) => {
                    *env = new_env;
                    *ctx = new_ctx;
                    lp += w.ln() - z.ln();
                    stack.extend(term.args().into_iter().zip(arg_types).rev());
                }
            }
        }
        Ok(lp)
    }
    fn logprior_rule(
        &self,
        rule: &Rule,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        ctx: &mut TypeContext,
    ) -> Result<f64, SampleError> {
        let mut env = Environment::from_vars(&rule.variables(), ctx);
        let schema = self.infer_rule(rule, &mut HashMap::new(), &mut env, ctx)?;
        let tp = schema.instantiate(ctx);
        let mut env = Environment::new(invent);
        let lp_lhs = self.logprior_term_internal(&rule.lhs, &tp, atom_weights, &mut env, ctx)?;
        let mut lp = 0.0;
        env.invent = false;
        for rhs in &rule.rhs {
            lp += lp_lhs + self.logprior_term_internal(&rhs, &tp, atom_weights, &mut env, ctx)?;
        }
        Ok(lp)
    }
    fn logprior_utrs<F>(
        &self,
        utrs: &UntypedTRS,
        p_of_n_rules: F,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        ctx: &mut TypeContext,
    ) -> Result<f64, SampleError>
    where
        F: Fn(usize) -> f64,
    {
        let clauses = utrs.clauses();
        let mut p_clauses = 0.0;
        for clause in &clauses {
            p_clauses += self.logprior_rule(clause, atom_weights, invent, &mut ctx.clone())?;
        }
        Ok(p_of_n_rules(clauses.len()) + p_clauses)
    }
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::too_many_arguments))]
    fn logprior_string_rule(
        &self,
        rule: &Rule,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
        ctx: &mut TypeContext,
    ) -> Result<f64, SampleError> {
        let mut env = Environment::from_vars(&rule.variables(), ctx);
        let schema = self.infer_rule(rule, &mut HashMap::new(), &mut env, ctx)?;
        let tp = schema.instantiate(ctx);
        let mut env = Environment::new(invent);
        let lp_lhs = self.logprior_term_internal(&rule.lhs, &tp, atom_weights, &mut env, ctx)?;
        let mut lp = 0.0;
        for rhs in &rule.rhs {
            lp += lp_lhs
                + UntypedTRS::p_string(&rule.lhs, rhs, dist, t_max, d_max, &self.signature)
                    .ok_or(SampleError::Subterm)?
        }
        Ok(lp)
    }
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::too_many_arguments))]
    fn logprior_srs<F>(
        &self,
        utrs: &UntypedTRS,
        p_of_n_rules: F,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
        ctx: &mut TypeContext,
    ) -> Result<f64, SampleError>
    where
        F: Fn(usize) -> f64,
    {
        let clauses = utrs.clauses();
        let mut p_clauses = 0.0;
        for clause in &clauses {
            p_clauses +=
                self.logprior_string_rule(clause, atom_weights, invent, dist, t_max, d_max, ctx)?;
        }
        Ok(p_of_n_rules(clauses.len()) + p_clauses)
    }
}
impl<'a, 'b, 'c> From<&'c TRSGP<'a, 'b>> for &'c Lexicon<'b> {
    fn from(gp_lex: &'c TRSGP<'a, 'b>) -> &'c Lexicon<'b> {
        &gp_lex.lexicon
    }
}

impl GenerationLimit {
    pub fn is_okay(&self, size: &[usize]) -> bool {
        match *self {
            GenerationLimit::TotalSize(n) => n >= size.iter().sum(),
            GenerationLimit::TermSize(n) => n >= *size.iter().max().unwrap(),
        }
    }
}

impl<'a> RuleEnumeration<'a> {
    fn new(
        lex: &'a Lex,
        schema: &TypeSchema,
        limit: GenerationLimit,
        invent: bool,
        ctx: &TypeContext,
    ) -> Self {
        let mut ctx = ctx.clone();
        let tp = schema.instantiate(&mut ctx);
        let env = Environment::new(invent);
        RuleEnumeration {
            stack: vec![(
                RuleContext::default(),
                (1, 1),
                vec![tp.clone(), tp],
                env,
                ctx,
            )],
            limit,
            lex,
            invent,
        }
    }
}
impl<'a> Iterator for RuleEnumeration<'a> {
    type Item = Rule;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((partial, (lsize, rsize), arg_types, mut env, mut ctx)) = self.stack.pop() {
            match Rule::try_from(&partial) {
                Ok(rule) => return Some(rule),
                Err(hole_place) => {
                    let lhs_hole = hole_place[0] == 0;
                    env.invent = self.invent && lhs_hole && hole_place != [0];
                    for (atom, mut new_arg_types, new_env, new_ctx) in
                        self.lex.valid_atoms(&arg_types[0], &env, &mut ctx)
                    {
                        let subcontext = Context::from(atom);
                        let new_size = if lhs_hole {
                            (lsize + subcontext.size() - 1, rsize)
                        } else {
                            (lsize, rsize + subcontext.size() - 1)
                        };
                        if self.limit.is_okay(&[new_size.0, new_size.1]) {
                            let new_context = partial.replace(&hole_place, subcontext).unwrap();
                            new_arg_types.extend_from_slice(&arg_types[1..]);
                            self.stack.push((
                                new_context,
                                new_size,
                                new_arg_types,
                                new_env,
                                new_ctx,
                            ));
                        }
                    }
                }
            }
        }
        None
    }
}

impl<'a> TermEnumeration<'a> {
    fn new(
        lex: &'a Lex,
        schema: &TypeSchema,
        max_size: usize,
        env: &Environment,
        ctx: &TypeContext,
    ) -> Self {
        let mut ctx = ctx.clone();
        let tp = schema.instantiate(&mut ctx);
        let limit = GenerationLimit::TotalSize(max_size);
        TermEnumeration {
            stack: vec![(Context::Hole, 1, vec![tp], env.clone(), ctx)],
            limit,
            lex,
        }
    }
}
impl<'a> Iterator for TermEnumeration<'a> {
    type Item = Term;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((partial, size, arg_types, env, mut ctx)) = self.stack.pop() {
            match Term::try_from(&partial) {
                Ok(term) => return Some(term),
                Err(hole_place) => {
                    // Find the options fitting the hole:
                    for (atom, mut new_arg_types, new_env, new_ctx) in
                        self.lex.valid_atoms(&arg_types[0], &env, &mut ctx)
                    {
                        let subcontext = Context::from(atom);
                        let new_size = size + subcontext.size() - 1;
                        if self.limit.is_okay(&[new_size]) {
                            let new_context = partial.replace(&hole_place, subcontext).unwrap();
                            new_arg_types.extend_from_slice(&arg_types[1..]);
                            self.stack.push((
                                new_context,
                                new_size,
                                new_arg_types,
                                new_env,
                                new_ctx,
                            ));
                        }
                    }
                }
            }
        }
        None
    }
}

use super::{SampleError, TRSMoveName, TRSMoves, TypeError, TRS};
use itertools::Itertools;
use polytype::{Context as TypeContext, Type, TypeSchema, Variable as TypeVar};
use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng,
};
use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::HashMap,
    f64::NEG_INFINITY,
    fmt,
    sync::{Arc, RwLock},
};
use term_rewriting::{
    Atom, Context, MergeStrategy, Operator, PStringDist, Place, Rule, RuleContext, Signature,
    SignatureChange, Term, Variable, TRS as UntypedTRS,
};
use utils::{logsumexp, weighted_sample};
use {Tournament, GP};

type LOpt = (Option<Atom>, Vec<Type>);

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Parameters for [`Lexicon`] genetic programming ([`GP`]).
///
/// [`Lexicon`]: struct.Lexicon.html
/// [`GP`]: ../trait.GP.html
pub struct GeneticParams {
    // A list of the moves available during search.
    pub moves: TRSMoves,
    /// The maximum number of nodes a sampled `Term` can have without failing.
    pub max_sample_size: usize,
    /// The weight to assign variables, constants, and non-constant operators, respectively.
    pub atom_weights: (f64, f64, f64, f64),
    /// `true` if you want only deterministic TRSs during search, else `false`.
    pub deterministic: bool,
}

pub struct ContextPoint<'a, O, E> {
    result: Result<O, E>,
    snapshot: usize,
    lex: Lexicon<'a>,
}
impl<'a, O, E> ContextPoint<'a, O, E> {
    pub fn drop(self) -> Result<O, E> {
        self.lex
            .0
            .ctx
            .write()
            .expect("poisoned context")
            .rollback(self.snapshot);
        self.result
    }
    pub fn keep(self) -> Result<O, E> {
        self.result
    }
    pub fn defer(self) -> (Result<O, E>, usize) {
        (self.result, self.snapshot)
    }
}

/// (representation) Manages the syntax of a term rewriting system.
#[derive(Clone)]
pub struct Lexicon<'a>(pub(crate) Cow<'a, Lex>);
impl<'a> Eq for Lexicon<'a> {}
impl<'a> Lexicon<'a> {
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
        for (id, name, tp) in operators {
            signature.new_op(id, name);
            ops.push(tp);
        }
        let mut lex = Lexicon(Cow::Owned(Lex {
            ops,
            vars: vec![],
            free_vars: vec![],
            signature,
            ctx: Arc::new(RwLock::new(ctx)),
        }));
        lex.0.to_mut().recompute_free_vars();
        lex
    }
    /// Convert a [`term_rewriting::Signature`] into a `Lexicon`:
    /// - `ops` are types for the [`term_rewriting::Operator`]s
    /// - `vars` are types for the [`term_rewriting::Variable`]s,
    /// - `background` are [`term_rewriting::Rule`]s that never change
    /// - `templates` are [`term_rewriting::RuleContext`]s that can serve as templates during learning
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
    /// # use term_rewriting::{Signature, parse_rule, parse_rulecontext};
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
    /// let vars = vec![];
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, TypeContext::default());
    /// ```
    ///
    /// [`polytype::ptp`]: https://docs.rs/polytype/~6.0/polytype/macro.ptp.html
    /// [`polytype::TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`term_rewriting::Signature`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Signature.html
    /// [`term_rewriting::Operator`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Operator.html
    /// [`term_rewriting::Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
    /// [`term_rewriting::Variable`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Variable.html
    /// [`term_rewriting::RuleContext`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.RuleContext.html
    pub fn from_signature<'b>(
        signature: Signature,
        ops: Vec<TypeSchema>,
        vars: Vec<TypeSchema>,
        ctx: TypeContext,
    ) -> Lexicon<'b> {
        let mut lex = Lexicon(Cow::Owned(Lex {
            ops,
            vars,
            free_vars: vec![],
            signature,
            ctx: Arc::new(RwLock::new(ctx)),
        }));
        lex.0.to_mut().recompute_free_vars();
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
        let mut ctx = lex1.0.ctx.read().expect("poisoned context").clone();
        // TODO: Are these the right things to share?
        let sacreds = lex2
            .0
            .free_vars
            .iter()
            .filter(|fv| lex1.0.free_vars.contains(fv))
            .cloned()
            .collect_vec();
        let ctx_change = ctx.merge(
            lex2.0.ctx.read().expect("poisoned context").clone(),
            sacreds,
        );
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
        // 4. Update vars.
        let mut vars = lex1.0.vars.clone();
        for schema in &lex2.0.vars {
            let mut schema = schema.clone();
            ctx_change.reify_typeschema(&mut schema);
            vars.push(schema);
        }
        // 5. Create and return a new Lexicon from parts.
        Ok((Lexicon::from_signature(sig, ops, vars, ctx), sig_change))
    }
    /// Return the specified operator if possible.
    pub fn has_op(&self, name: Option<&str>, arity: u8) -> Result<Operator, SampleError> {
        let sig = &self.0.signature;
        sig.operators()
            .into_iter()
            .find(|op| {
                op.arity() == arity
                    && op.name(sig).as_ref().map(std::string::String::as_str) == name
            })
            .ok_or(SampleError::OptionsExhausted)
    }
    /// All the [`term_rewriting::Variable`]s in the `Lexicon`.
    ///
    /// [`term_rewriting::Variable`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Variable.html
    pub fn variables(&self) -> Vec<Variable> {
        self.0.variables()
    }
    /// All the free type variables in the `Lexicon`.
    pub fn free_vars(&self) -> Vec<TypeVar> {
        self.0.free_vars().to_vec()
    }
    /// Add a new variable to the `Lexicon`.
    pub fn invent_variable(&mut self, tp: &Type) -> Variable {
        self.0.to_mut().invent_variable(tp)
    }
    /// Add a new operator to the `Lexicon`.
    pub fn invent_operator(&mut self, name: Option<String>, arity: u8, tp: &Type) -> Operator {
        self.0.to_mut().invent_operator(name, arity, tp)
    }
    /// Shrink the universe of symbols.
    pub fn contract(&mut self, ids: &[usize]) -> usize {
        self.0.to_mut().contract(ids)
    }
    /// Return the `Lexicon`'s [`TypeContext`].
    ///
    /// [`TypeContext`]: https://docs.rs/polytype/~6.0/polytype/struct.Context.html
    pub fn context(&self) -> TypeContext {
        self.0.ctx.read().expect("poisoned context").clone()
    }
    /// Return the `Lexicon`'s `Signature`.
    pub fn signature(&self) -> Signature {
        self.0.signature.clone()
    }
    pub fn snapshot(&self) -> usize {
        self.0.snapshot()
    }
    pub fn rollback(&self, snapshot: usize) {
        self.0.rollback(snapshot);
    }
    pub fn instantiate(&self, schema: &TypeSchema) -> Type {
        schema.instantiate(&mut self.0.ctx.write().expect("poisoned context"))
    }
    pub fn unify(&self, t1: &Type, t2: &Type) -> Result<(), ()> {
        self.0.unify(t1, t2).map_err(|_| ())
    }
    /// Return a new [`Type::Variable`] from the `Lexicon`'s [`TypeContext`].
    ///
    /// [`Type::Variable`]: https://docs.rs/polytype/~6.0/polytype/enum.Type.html
    /// [`TypeContext`]: https://docs.rs/polytype/~6.0/polytype/struct.Context.html
    pub fn fresh_type_variable(&self) -> Type {
        self.0.ctx.write().expect("poisoned context").new_variable()
    }
    /// Infer the [`polytype::TypeSchema`] associated with a [`term_rewriting::Context`].
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::Lexicon;
    /// # use term_rewriting::{Context, Signature, parse_rule};
    /// # use polytype::Context as TypeContext;
    /// # use std::collections::HashMap;
    /// let mut sig = Signature::default();
    ///
    /// let vars = vec![];
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some("PLUS".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// let succ = sig.new_op(1, Some("SUCC".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    /// sig.new_op(0, Some("ZERO".to_string()));
    /// ops.push(ptp![int]);
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, TypeContext::default());
    ///
    /// let context = Context::Application {
    ///     op: succ,
    ///     args: vec![Context::Hole]
    /// };
    /// let mut ctx = lexicon.context();
    ///
    /// let inferred_schema = lexicon.infer_context(&context, &mut HashMap::new()).drop().unwrap();
    ///
    /// assert_eq!(inferred_schema, ptp![int]);
    /// ```
    ///
    /// [`polytype::TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`term_rewriting::Context`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Context.html
    pub fn infer_context(
        &self,
        context: &Context,
        types: &mut HashMap<Place, Type>,
    ) -> ContextPoint<TypeSchema, TypeError> {
        let snapshot = self.snapshot();
        let result = self.0.infer_context(context, types);
        for (_, v) in types.iter_mut() {
            v.apply_mut_compress(&mut self.0.ctx.write().expect("poisoned context"));
        }
        ContextPoint {
            snapshot,
            result,
            lex: self.clone(),
        }
    }
    /// Infer the `TypeSchema` associated with a `RuleContext`.
    pub fn infer_rulecontext(&self, context: &RuleContext) -> Result<TypeSchema, TypeError> {
        let snapshot = self.snapshot();
        let result = self.0.infer_rulecontext(context);
        self.rollback(snapshot);
        result
    }
    /// Infer the `TypeSchema` associated with a `Rule`.
    pub fn infer_rule(
        &self,
        rule: &Rule,
        types: &mut HashMap<Place, Type>,
    ) -> ContextPoint<TypeSchema, TypeError> {
        let snapshot = self.snapshot();
        let result = self.0.infer_rule(rule, types);
        for (_, v) in types.iter_mut() {
            v.apply_mut_compress(&mut self.0.ctx.write().expect("poisoned lexicon"));
        }
        ContextPoint {
            snapshot,
            result,
            lex: self.clone(),
        }
    }
    /// Infer the `TypeSchema` associated with a collection of `Rules`.
    pub fn infer_rules(&self, rules: &[Rule]) -> Result<TypeSchema, TypeError> {
        let snapshot = self.snapshot();
        let result = self.0.infer_rules(rules);
        self.rollback(snapshot);
        result
    }
    /// Infer the `TypeSchema` associated with an `Operator`.
    pub fn infer_op(&self, op: Operator) -> Result<TypeSchema, TypeError> {
        let snapshot = self.snapshot();
        let result = self.0.op_tp(op).map(|tp| tp.clone());
        self.rollback(snapshot);
        result
    }
    /// Infer the `TypeSchema` associated with a `Variable`.
    pub fn infer_var(&self, var: Variable) -> Result<TypeSchema, TypeError> {
        let snapshot = self.snapshot();
        let result = self.0.var_tp(var).map(|tp| tp.clone());
        self.rollback(snapshot);
        result
    }
    /// Infer the `TypeSchema` associated with a `Term`.
    pub fn infer_term(
        &self,
        term: &Term,
        types: &mut HashMap<Place, Type>,
    ) -> ContextPoint<TypeSchema, TypeError> {
        let snapshot = self.snapshot();
        let result = self.0.infer_term(term, types);
        for (_, v) in types.iter_mut() {
            v.apply_mut_compress(&mut self.0.ctx.write().expect("poisoned context"));
        }
        ContextPoint {
            snapshot,
            result,
            lex: self.clone(),
        }
    }
    /// Infer the `TypeSchema` associated with a `TRS`.
    pub fn infer_utrs(&self, utrs: &UntypedTRS) -> Result<(), TypeError> {
        let snapshot = self.snapshot();
        let result = self.0.infer_utrs(utrs);
        self.rollback(snapshot);
        result
    }
    /// Sample a [`term_rewriting::Term`].
    ///
    /// # Example
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
    /// let mut lexicon = Lexicon::new(operators, TypeContext::default());
    ///
    /// let schema = ptp![int];
    /// let mut ctx = lexicon.context();
    /// let invent = true;
    /// let variable = true;
    /// let atom_weights = (1.0, 1.0, 1.0, 1.0);
    /// let max_size = 50;
    /// let mut vars = vec![];
    ///
    /// let term = lexicon.sample_term(&schema, atom_weights, invent, variable, max_size, &mut vars).unwrap();
    /// ```
    ///
    /// [`term_rewriting::Term`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Term.html
    pub fn sample_term(
        &mut self,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        variable: bool,
        max_size: usize,
        vars: &mut Vec<Variable>,
    ) -> Result<Term, SampleError> {
        let snapshot = self.snapshot();
        let result =
            self.0
                .to_mut()
                .sample_term(schema, atom_weights, invent, variable, max_size, vars);
        self.rollback(snapshot);
        result
    }
    /// Sample a `Term` conditioned on a `Context` rather than a `TypeSchema`.
    pub fn sample_term_from_context(
        &mut self,
        context: &Context,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        variable: bool,
        max_size: usize,
    ) -> Result<Term, SampleError> {
        let snapshot = self.snapshot();
        let result = self.0.to_mut().sample_term_from_context(
            context,
            atom_weights,
            invent,
            variable,
            max_size,
        );
        self.rollback(snapshot);
        result
    }
    /// Sample a `Rule`.
    pub fn sample_rule(
        &mut self,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        max_size: usize,
    ) -> Result<Rule, SampleError> {
        let snapshot = self.snapshot();
        let result = self
            .0
            .to_mut()
            .sample_rule(schema, atom_weights, invent, max_size);
        self.rollback(snapshot);
        result
    }
    /// Sample a `Rule` conditioned on a `Context` rather than a `TypeSchema`.
    pub fn sample_rule_from_context(
        &mut self,
        context: RuleContext,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        max_size: usize,
    ) -> ContextPoint<Rule, SampleError> {
        let snapshot = self.snapshot();
        let result =
            self.0
                .to_mut()
                .sample_rule_from_context(context, atom_weights, invent, max_size);
        ContextPoint {
            snapshot,
            result,
            lex: self.clone(),
        }
    }
    pub fn enumerate_to_n_terms(
        &mut self,
        schema: &TypeSchema,
        invent: bool,
        associative: bool,
        n: usize,
    ) -> Vec<Term> {
        let snapshot = self.snapshot();
        let vec = self
            .0
            .to_mut()
            .enumerate_to_n_terms(schema, invent, associative, n);
        self.rollback(snapshot);
        vec
    }
    pub fn enumerate_n_terms(
        &mut self,
        schema: &TypeSchema,
        invent: bool,
        associative: bool,
        n: usize,
    ) -> Vec<Term> {
        let snapshot = self.snapshot();
        let vec = self
            .0
            .to_mut()
            .enumerate_n_terms(schema, invent, associative, n);
        self.rollback(snapshot);
        vec
    }
    pub fn enumerate_to_n_rules(
        &mut self,
        schema: &TypeSchema,
        invent: bool,
        associative: bool,
        n: usize,
    ) -> Vec<Rule> {
        let snapshot = self.snapshot();
        let vec = self
            .0
            .to_mut()
            .enumerate_to_n_rules(schema, invent, associative, n);
        self.rollback(snapshot);
        vec
    }
    pub fn enumerate_n_rules(
        &mut self,
        schema: &TypeSchema,
        invent: bool,
        associative: bool,
        n: usize,
    ) -> Vec<Rule> {
        let snapshot = self.snapshot();
        let vec = self
            .0
            .to_mut()
            .enumerate_n_rules(schema, invent, associative, n);
        self.rollback(snapshot);
        vec
    }
    /// Give the log probability of sampling a Term.
    pub fn logprior_term(
        &self,
        term: &Term,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        let snapshot = self.snapshot();
        let result = self.0.logprior_term(term, schema, atom_weights, invent);
        self.rollback(snapshot);
        result
    }
    /// Give the log probability of sampling a Rule.
    pub fn logprior_rule(
        &self,
        rule: &Rule,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        let snapshot = self.snapshot();
        let result = self.0.logprior_rule(rule, atom_weights, invent);
        self.rollback(snapshot);
        result
    }
    /// Give the log probability of sampling a TRS.
    pub fn logprior_utrs(
        &self,
        utrs: &UntypedTRS,
        p_number_of_rules: Box<dyn Fn(usize) -> f64>,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        let snapshot = self.snapshot();
        let result = self
            .0
            .logprior_utrs(utrs, p_number_of_rules, atom_weights, invent);
        self.rollback(snapshot);
        result
    }
    /// Give the log probability of sampling an SRS.
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::too_many_arguments))]
    pub fn logprior_srs(
        &self,
        utrs: &UntypedTRS,
        p_number_of_rules: Box<dyn Fn(usize) -> f64>,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
    ) -> Result<f64, SampleError> {
        let snapshot = self.snapshot();
        let result = self.0.logprior_srs(
            utrs,
            p_number_of_rules,
            atom_weights,
            invent,
            dist,
            t_max,
            d_max,
        );
        self.rollback(snapshot);
        result
    }
}
impl<'a> fmt::Debug for Lexicon<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Lexicon({:?})", self.0)
    }
}
impl<'a> PartialEq for Lexicon<'a> {
    fn eq(&self, other: &Lexicon) -> bool {
        self.0 == other.0
    }
}
impl<'a> fmt::Display for Lexicon<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug)]
pub(crate) struct Lex {
    pub(crate) ops: Vec<TypeSchema>,
    pub(crate) vars: Vec<TypeSchema>,
    free_vars: Vec<TypeVar>,
    pub(crate) signature: Signature,
    pub(crate) ctx: Arc<RwLock<TypeContext>>,
}
impl PartialEq for Lex {
    fn eq(&self, other: &Self) -> bool {
        self.ops == other.ops && self.vars == other.vars && self.signature == other.signature
    }
}
impl Eq for Lex {}
impl std::clone::Clone for Lex {
    fn clone(&self) -> Self {
        Lex {
            ops: self.ops.clone(),
            vars: self.vars.clone(),
            free_vars: self.free_vars.clone(),
            // TODO: confirm that we don't need a deep copy here.
            signature: self.signature.deep_copy(),
            ctx: Arc::new(RwLock::new(
                self.ctx.read().expect("poisoned context").clone(),
            )),
        }
    }
}
impl fmt::Display for Lex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Signature:")?;
        for (op, schema) in self.signature.operators().iter().zip(&self.ops) {
            writeln!(
                f,
                "{}/{}: {}",
                op.display(&self.signature),
                op.arity(),
                schema
            )?;
        }
        for (var, schema) in self.signature.variables().iter().zip(&self.vars) {
            writeln!(f, "{}: {}", var.display(&self.signature), schema)?;
        }
        Ok(())
    }
}
impl Lex {
    fn variables(&self) -> Vec<Variable> {
        self.signature.variables()
    }
    fn free_vars(&self) -> &[TypeVar] {
        &self.free_vars
    }
    fn free_vars_applied(&self) -> Vec<TypeVar> {
        let mut vars = vec![];
        for x in &self.free_vars {
            let mut v = Type::Variable(*x);
            v.apply_mut_compress(&mut self.ctx.write().expect("poisoned context"));
            vars.append(&mut v.vars());
        }
        vars.sort();
        vars.dedup();
        vars
    }
    fn invent_variable(&mut self, tp: &Type) -> Variable {
        let var = self.signature.new_var(None);
        self.vars.push(TypeSchema::Monotype(tp.clone()));
        self.free_vars.append(&mut tp.vars());
        self.free_vars.sort_unstable();
        self.free_vars.dedup();
        var
    }
    fn invent_operator(&mut self, name: Option<String>, arity: u8, tp: &Type) -> Operator {
        let op = self.signature.new_op(arity, name);
        self.ops.push(TypeSchema::Monotype(tp.clone()));
        self.free_vars.append(&mut tp.vars());
        self.free_vars.sort_unstable();
        self.free_vars.dedup();
        op
    }
    fn contract(&mut self, ids: &[usize]) -> usize {
        let n = self.signature.contract(ids);
        self.ops.truncate(n + 1);
        self.recompute_free_vars();
        n
    }
    fn snapshot(&self) -> usize {
        self.ctx.read().expect("poisoned context").len()
    }
    fn rollback(&self, snapshot: usize) {
        self.ctx
            .write()
            .expect("poisoned context")
            .rollback(snapshot);
    }
    fn unify(&self, t1: &Type, t2: &Type) -> Result<(), TypeError> {
        self.ctx
            .write()
            .expect("poisoned context")
            .unify(t1, t2)
            .map_err(TypeError::from)
    }
    fn recompute_free_vars(&mut self) -> &[TypeVar] {
        self.free_vars.clear();
        for op in &self.ops {
            self.free_vars.append(&mut op.free_vars())
        }
        for var in &self.vars {
            self.free_vars.append(&mut var.free_vars())
        }
        self.free_vars.sort_unstable();
        self.free_vars.dedup();
        &self.free_vars
    }
    fn fit_atom(&self, atom: &Atom, tp: &Type, rollback: bool) -> Result<Vec<Type>, SampleError> {
        let atom_tp = self.instantiate_atom(atom)?;
        let rollback_n = self.snapshot();
        let unify_tp = match atom {
            Atom::Operator(o) if o.arity() > 0 => atom_tp.returns().ok_or(TypeError::Malformed)?,
            _ => &atom_tp,
        };
        let result = self.unify(unify_tp, tp);
        if rollback {
            self.rollback(rollback_n);
        }
        result
            .map(|_| match atom {
                Atom::Operator(o) if o.arity() > 0 => {
                    atom_tp.args_destruct().unwrap_or_else(Vec::new)
                }
                _ => Vec::with_capacity(0),
            })
            .map_err(SampleError::from)
    }
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::too_many_arguments))]
    fn place_atom(
        &mut self,
        atom: &Atom,
        arg_types: Vec<Type>,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        max_size: usize,
        vars: &mut Vec<Variable>,
    ) -> Result<Term, SampleError> {
        match *atom {
            Atom::Variable(v) => Ok(Term::Variable(v)),
            Atom::Operator(op) => {
                let mut size = 1;
                let snapshot = self.snapshot(); // for "undo" semantics
                let mut args = Vec::with_capacity(arg_types.len());
                let can_be_variable = true; // subterms can always be variables
                for arg_tp in arg_types {
                    let subtype =
                        arg_tp.apply_compress(&mut self.ctx.write().expect("poisoned context"));
                    let lex_vars = self.free_vars_applied();
                    let arg_schema = subtype.generalize(&lex_vars);
                    if size > max_size {
                        return Err(SampleError::SizeExceeded(size, max_size));
                    }
                    let result = self
                        .sample_term_internal(
                            &arg_schema,
                            atom_weights,
                            invent,
                            can_be_variable,
                            max_size - size,
                            vars,
                        )
                        .map_err(|_| SampleError::Subterm)
                        .and_then(|subterm| {
                            let tp = self
                                .infer_term(&subterm, &mut HashMap::new())?
                                .instantiate_owned(
                                    &mut self.ctx.write().expect("poisoned context"),
                                );
                            self.ctx
                                .write()
                                .expect("poisoned context")
                                .unify_fast(subtype, tp)?;
                            Ok(subterm)
                        });
                    match result {
                        Ok(subterm) => {
                            size += subterm.size();
                            args.push(subterm);
                        }
                        Err(e) => {
                            self.rollback(snapshot);
                            return Err(e);
                        }
                    }
                }
                Ok(Term::Application { op, args })
            }
        }
    }
    fn instantiate_atom(&self, atom: &Atom) -> Result<Type, TypeError> {
        let schema = self.infer_atom(atom)?.clone();
        let mut tp = schema.instantiate_owned(&mut self.ctx.write().expect("poisoned context"));
        tp.apply_mut_compress(&mut self.ctx.write().expect("poisoned context"));
        Ok(tp)
    }
    fn var_tp(&self, v: Variable) -> Result<&TypeSchema, TypeError> {
        let v_id = v.id();
        if v_id >= self.vars.len() {
            Err(TypeError::NotFound)
        } else {
            Ok(&self.vars[v_id])
        }
    }
    fn op_tp(&self, o: Operator) -> Result<&TypeSchema, TypeError> {
        let o_id = o.id();
        if o_id >= self.ops.len() {
            Err(TypeError::NotFound)
        } else {
            Ok(&self.ops[o_id])
        }
    }

    fn infer_atom(&self, atom: &Atom) -> Result<&TypeSchema, TypeError> {
        match *atom {
            Atom::Operator(ref o) => self.op_tp(o.clone()),
            Atom::Variable(ref v) => self.var_tp(v.clone()),
        }
    }
    fn infer_term(
        &self,
        term: &Term,
        tps: &mut HashMap<Place, Type>,
    ) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_term_internal(term, vec![], tps)?;
        let lex_vars = self.free_vars_applied();
        Ok(tp
            .apply_compress(&mut self.ctx.write().expect("poisoned context"))
            .generalize(&lex_vars))
    }
    fn infer_term_internal(
        &self,
        term: &Term,
        place: Place,
        tps: &mut HashMap<Place, Type>,
    ) -> Result<Type, TypeError> {
        let tp = match *term {
            Term::Variable(v) => self.instantiate_atom(&Atom::from(v))?,
            Term::Application { op, ref args } => {
                let head_type = self.instantiate_atom(&Atom::from(op))?;
                let body_type = {
                    let mut pre_types = Vec::with_capacity(args.len() + 1);
                    for (i, a) in args.iter().enumerate() {
                        let mut new_place = place.clone();
                        new_place.push(i);
                        pre_types.push(self.infer_term_internal(a, new_place, tps)?);
                    }
                    pre_types.push(self.ctx.write().expect("poisoned context").new_variable());
                    Type::from(pre_types)
                };
                self.unify(&head_type, &body_type)?;
                if op.arity() > 0 {
                    head_type
                        .returns()
                        .unwrap_or(&head_type)
                        .apply_compress(&mut self.ctx.write().expect("poisoned context"))
                } else {
                    head_type.apply_compress(&mut self.ctx.write().expect("poisoned context"))
                }
            }
        };
        tps.insert(place, tp.clone());
        Ok(tp)
    }
    fn infer_context(
        &self,
        context: &Context,
        tps: &mut HashMap<Place, Type>,
    ) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_context_internal(context, vec![], tps)?;
        let lex_vars = self.free_vars_applied();
        Ok(tp
            .apply_compress(&mut self.ctx.write().expect("poisoned context"))
            .generalize(&lex_vars))
    }
    fn infer_context_internal(
        &self,
        context: &Context,
        place: Place,
        tps: &mut HashMap<Place, Type>,
    ) -> Result<Type, TypeError> {
        let tp = match *context {
            Context::Hole => self.ctx.write().expect("poisoned context").new_variable(),
            Context::Variable(v) => self.instantiate_atom(&Atom::from(v))?,
            Context::Application { op, ref args } => {
                let head_type = self.instantiate_atom(&Atom::from(op))?;
                let body_type = {
                    let mut pre_types = Vec::with_capacity(args.len() + 1);
                    for (i, a) in args.iter().enumerate() {
                        let mut new_place = place.clone();
                        new_place.push(i);
                        pre_types.push(self.infer_context_internal(a, new_place, tps)?);
                    }
                    pre_types.push(self.ctx.write().expect("poisoned context").new_variable());
                    Type::from(pre_types)
                };
                self.unify(&head_type, &body_type)?;
                if op.arity() == 0 {
                    head_type.apply_compress(&mut self.ctx.write().expect("poisoned context"))
                } else {
                    head_type
                        .returns()
                        .unwrap_or(&head_type)
                        .apply_compress(&mut self.ctx.write().expect("poisoned context"))
                }
            }
        };
        tps.insert(place, tp.clone());
        Ok(tp)
    }
    fn infer_rule(
        &self,
        rule: &Rule,
        tps: &mut HashMap<Place, Type>,
    ) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_rule_internal(rule, tps)?;
        let lex_vars = self.free_vars_applied();
        Ok(tp
            .apply_compress(&mut self.ctx.write().expect("poisoned context"))
            .generalize(&lex_vars))
    }
    fn infer_rule_internal(
        &self,
        rule: &Rule,
        tps: &mut HashMap<Place, Type>,
    ) -> Result<Type, TypeError> {
        let lhs_type = self.infer_term_internal(&rule.lhs, vec![0], tps)?;
        let rhs_types = rule
            .rhs
            .iter()
            .enumerate()
            .map(|(i, rhs)| self.infer_term_internal(&rhs, vec![i + 1], tps))
            .collect::<Result<Vec<Type>, _>>()?;
        // unify to introduce rule-level constraints
        for rhs_type in rhs_types {
            self.unify(&lhs_type, &rhs_type)?;
        }
        Ok(lhs_type.apply_compress(&mut self.ctx.write().expect("poisoned context")))
    }
    fn infer_rules(&self, rules: &[Rule]) -> Result<TypeSchema, TypeError> {
        let rule_tps = rules
            .iter()
            .map(|rule| {
                self.infer_rule(rule, &mut HashMap::new()).map(|rule_tp| {
                    rule_tp.instantiate(&mut self.ctx.write().expect("poisoned context"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tp = self.ctx.write().expect("poisoned context").new_variable();
        for rule_tp in rule_tps {
            self.unify(&tp, &rule_tp)?;
        }
        let lex_vars = self.free_vars_applied();
        Ok(tp
            .apply_compress(&mut self.ctx.write().expect("poisoned context"))
            .generalize(&lex_vars))
    }
    fn infer_rulecontext(&self, context: &RuleContext) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_rulecontext_internal(context, &mut HashMap::new())?;
        let lex_vars = self.free_vars_applied();
        Ok(tp
            .apply_compress(&mut self.ctx.write().expect("poisoned context"))
            .generalize(&lex_vars))
    }
    fn infer_rulecontext_internal(
        &self,
        context: &RuleContext,
        tps: &mut HashMap<Place, Type>,
    ) -> Result<Type, TypeError> {
        let lhs_type = self.infer_context_internal(&context.lhs, vec![0], tps)?;
        let rhs_types = context
            .rhs
            .iter()
            .enumerate()
            .map(|(i, rhs)| self.infer_context_internal(&rhs, vec![i + 1], tps))
            .collect::<Result<Vec<Type>, _>>()?;
        // unify to introduce rule-level constraints
        for rhs_type in rhs_types {
            self.unify(&lhs_type, &rhs_type)?;
        }
        Ok(lhs_type.apply_compress(&mut self.ctx.write().expect("poisoned context")))
    }
    fn infer_utrs(&self, utrs: &UntypedTRS) -> Result<(), TypeError> {
        for rule in &utrs.rules {
            self.infer_rule_internal(rule, &mut HashMap::new())?;
        }
        Ok(())
    }
    fn sample_term(
        &mut self,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        variable: bool,
        max_size: usize,
        vars: &mut Vec<Variable>,
    ) -> Result<Term, SampleError> {
        self.sample_term_internal(schema, atom_weights, invent, variable, max_size, vars)
    }
    fn sample_term_internal(
        &mut self,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        variable: bool,
        max_size: usize,
        vars: &mut Vec<Variable>,
    ) -> Result<Term, SampleError> {
        let tp = schema.instantiate(&mut self.ctx.write().expect("poisoned context"));
        let (atom, arg_types) = self.prepare_option(vars, atom_weights, invent, variable, &tp)?;
        self.place_atom(&atom, arg_types, atom_weights, invent, max_size, vars)
    }
    fn prepare_option(
        &mut self,
        vars: &mut Vec<Variable>,
        (vw, cw, ow, iw): (f64, f64, f64, f64),
        invent: bool,
        variable: bool,
        tp: &Type,
    ) -> Result<(Atom, Vec<Type>), SampleError> {
        // create options
        let ops = self.signature.operators();
        let mut atoms: Vec<_> = ops.into_iter().map(|o| Some(Atom::Operator(o))).collect();
        if variable {
            atoms.extend(vars.to_vec().into_iter().map(|v| Some(Atom::Variable(v))));
            if invent {
                atoms.push(None);
            }
        }
        let options = atoms
            .into_iter()
            .filter_map(|a| match a {
                None => Some(None),
                Some(atom) => match self.fit_atom(&atom, &tp, true) {
                    Err(_) => None,
                    Ok(_) => Some(Some(atom)),
                },
            })
            .collect_vec();
        if options.is_empty() {
            return Err(SampleError::OptionsExhausted);
        }
        // normalize the weights
        let z = vw + cw + ow;
        let (vlp, clp, olp) = ((vw / z).ln(), (cw / z).ln(), (ow / z).ln());
        let zs: (f64, f64, f64) = options
            .iter()
            .fold((0.0, 0.0, 0.0), |(vz, cz, oz), o| match o {
                None => (vz + iw, cz, oz),
                Some(Atom::Variable(_)) => (vz + 1.0, cz, oz),
                Some(Atom::Operator(ref o)) if o.arity() == 0 => (vz, cz + 1.0, oz),
                Some(Atom::Operator(_)) => (vz, cz, oz + 1.0),
            });
        // create weights
        let weights: Vec<_> = options
            .iter()
            .map(|ref o| match o {
                None => vlp + iw.ln() - zs.0.ln(),
                Some(Atom::Variable(_)) => vlp - zs.0.ln(),
                Some(Atom::Operator(ref o)) if o.arity() == 0 => clp - zs.1.ln(),
                Some(Atom::Operator(_)) => olp - zs.2.ln(),
            })
            .map(|w| w.exp())
            .collect();
        // sample an option that typechecks
        let option = weighted_sample(&options, &weights);
        let atom = option.unwrap_or_else(|| {
            let new_var = self.invent_variable(tp);
            vars.push(new_var);
            Atom::Variable(new_var)
        });
        let arg_types = self.fit_atom(&atom, &tp, false)?;
        Ok((atom, arg_types))
    }
    fn sample_term_from_context(
        &mut self,
        context: &Context,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        variable: bool,
        max_size: usize,
    ) -> Result<Term, SampleError> {
        let mut size = context.size();
        if size > max_size {
            return Err(SampleError::SizeExceeded(size, max_size));
        }
        let mut map = HashMap::new();
        let context = context.clone();
        let hole_places = context.holes();
        self.infer_context_internal(&context, vec![], &mut map)?;
        let lex_vars = self.free_vars_applied();
        let mut context_vars = context.variables();
        for p in &hole_places {
            size = context.size();
            let schema = &map[p]
                .apply_compress(&mut self.ctx.write().expect("poisoned context"))
                .generalize(&lex_vars);
            let subterm = self.sample_term_internal(
                &schema,
                atom_weights,
                invent,
                variable,
                max_size - size + 1, // + 1 to fill the hole
                &mut context_vars,
            )?;
            context.replace(&p, Context::from(subterm));
        }
        context.to_term().or(Err(SampleError::Subterm))
    }
    fn sample_rule(
        &mut self,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        max_size: usize,
    ) -> Result<Rule, SampleError> {
        loop {
            let snapshot = self.snapshot();
            let mut vars = vec![];
            let lhs = self.sample_term_internal(
                schema,
                atom_weights,
                invent,
                false,
                max_size - 1,
                &mut vars,
            )?;
            let rhs = self.sample_term_internal(
                schema,
                atom_weights,
                false,
                true,
                max_size - lhs.size(),
                &mut vars,
            )?;
            if let Some(rule) = Rule::new(lhs, vec![rhs]) {
                return Ok(rule);
            } else {
                self.rollback(snapshot);
            }
        }
    }
    fn sample_rule_from_context(
        &mut self,
        mut context: RuleContext,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        max_size: usize,
    ) -> Result<Rule, SampleError> {
        let mut size = context.size();
        if size > max_size {
            return Err(SampleError::SizeExceeded(size, max_size));
        }
        let mut tps = HashMap::new();
        let hole_places = context.holes();
        let mut context_vars = context.variables();
        self.infer_rulecontext_internal(&context, &mut tps)?;
        for p in &hole_places {
            size = context.size();
            let schema = TypeSchema::Monotype(
                tps[p].apply_compress(&mut self.ctx.write().expect("poisoned context")),
            );
            let can_invent = p[0] == 0 && invent;
            let can_be_variable = p != &vec![0];
            let subterm = self.sample_term_internal(
                &schema,
                atom_weights,
                can_invent,
                can_be_variable,
                max_size + 1 - size, // + 1 to fill the hole
                &mut context_vars,
            )?;
            context = context
                .replace(&p, Context::from(subterm))
                .ok_or(SampleError::Subterm)?;
        }
        context.to_rule().or({ Err(SampleError::Subterm) })
    }
    fn enumerate_to_n_terms(
        &mut self,
        schema: &TypeSchema,
        invent: bool,
        associative: bool,
        n: usize,
    ) -> Vec<Term> {
        (0..=n)
            .flat_map(|n_i| self.enumerate_n_terms(schema, invent, associative, n_i))
            .collect_vec()
    }
    fn enumerate_n_terms(
        &mut self,
        schema: &TypeSchema,
        invent: bool,
        associative: bool,
        n: usize,
    ) -> Vec<Term> {
        self.enumerate_n_terms_internal(schema, invent, associative, n, &mut vec![])
    }
    fn enumerate_to_n_rules(
        &mut self,
        schema: &TypeSchema,
        invent: bool,
        associative: bool,
        n: usize,
    ) -> Vec<Rule> {
        (0..=n)
            .flat_map(|n_i| self.enumerate_n_rules(schema, invent, associative, n_i))
            .collect_vec()
    }
    fn enumerate_n_rules(
        &mut self,
        schema: &TypeSchema,
        invent: bool,
        associative: bool,
        n: usize,
    ) -> Vec<Rule> {
        if n == 0 {
            return vec![];
        }
        let snapshot = self.snapshot();
        let lhss = self.enumerate_to_n_terms(schema, invent, associative, n - 1);
        self.rollback(snapshot);
        lhss.iter()
            .flat_map(|lhs| {
                let snapshot = self.snapshot();
                let rhss = if let Ok(schema) = self.infer_term(lhs, &mut HashMap::new()) {
                    self.enumerate_n_terms_internal(
                        &schema,
                        false,
                        associative,
                        n - lhs.size(),
                        &mut lhs.variables(),
                    )
                } else {
                    vec![]
                };
                self.rollback(snapshot);
                rhss.into_iter()
                    .filter_map(|rhs| {
                        Rule::new(lhs.clone(), vec![rhs]).and_then(|rule| {
                            let snapshot = self.snapshot();
                            let result = self.infer_rule(&rule, &mut HashMap::new());
                            self.rollback(snapshot);
                            result.ok().map(|_| rule)
                        })
                    })
                    .collect_vec()
            })
            .collect_vec()
    }
    fn prepare_options(
        &mut self,
        schema: &TypeSchema,
        invent: bool,
        complex: bool,
        vars: &mut Vec<Variable>,
    ) -> Vec<Atom> {
        let tp = schema.instantiate(&mut self.ctx.write().expect("poisoned context"));
        let ops = self.signature.operators();
        let mut options: Vec<_> = ops
            .into_iter()
            .filter(|o| complex || o.arity() < 2)
            .map(|o| Some(Atom::Operator(o)))
            .collect();
        options.extend(vars.to_vec().into_iter().map(|v| Some(Atom::Variable(v))));
        if invent {
            options.push(None);
        }
        options
            .into_iter()
            .filter_map(|option| {
                let atom = option.unwrap_or_else(|| {
                    let new_var = self.invent_variable(&tp);
                    Atom::Variable(new_var)
                });
                match self.fit_atom(&atom, &tp, true) {
                    Ok(_arg_types) => Some(atom),
                    _ => None,
                }
            })
            .collect_vec()
    }
    fn prepare_prior_options(
        &self,
        tp: &Type,
        term: &Term,
        invent: bool,
        vars: &[Variable],
    ) -> (Vec<LOpt>, Vec<LOpt>, Vec<LOpt>) {
        let mut vs = vars
            .iter()
            .filter_map(|&v| {
                let atom = Atom::from(v);
                self.fit_atom(&atom, &tp, true)
                    .map(|arg_types| (Some(atom), arg_types))
                    .ok()
            })
            .collect_vec();
        if invent {
            // empty arg_types below because variables have no arguments
            if let Term::Variable(v) = *term {
                if !vars.contains(&v) {
                    vs.push((Some(Atom::Variable(v)), vec![]));
                } else {
                    vs.push((None, vec![]));
                }
            } else {
                vs.push((None, vec![]));
            }
        }
        let (mut cs, mut os) = (vec![], vec![]);
        for &op in &self.signature.operators() {
            let atom = Atom::Operator(op);
            if let Ok(arg_types) = self.fit_atom(&atom, &tp, true) {
                if op.arity() == 0 {
                    // arg_types is empty, but using it avoids allocation
                    cs.push((Some(atom), arg_types));
                } else {
                    os.push((Some(atom), arg_types));
                }
            }
        }
        (vs, cs, os)
    }
    fn atom_priors(
        invent: bool,
        (vw, cw, ow, iw): (f64, f64, f64, f64),
        (nv, nc, no): (usize, usize, usize),
    ) -> (f64, f64, f64, f64) {
        let z = vw + cw + ow;
        let (vp, cp, op) = (vw / z, cw / z, ow / z);
        let vlp = if invent && nv == 1 {
            // in this case, the only variable is invented, so no mass goes here
            NEG_INFINITY
        } else {
            vp.ln() - ((nv as f64) + (invent as usize as f64) * (-1.0 + iw)).ln()
        };
        let ilp = if invent {
            vp.ln() + iw.ln() - ((nv as f64) - 1.0 + iw).ln()
        } else {
            NEG_INFINITY
        };
        let clp = if nc == 0 {
            NEG_INFINITY
        } else {
            cp.ln() - (nc as f64).ln()
        };
        let olp = if no == 0 {
            NEG_INFINITY
        } else {
            op.ln() - (no as f64).ln()
        };
        let mut lps = vec![];
        for i in 0..nv {
            if invent && i == 0 {
                lps.push(ilp);
            } else {
                lps.push(vlp);
            }
        }
        for _ in 0..nc {
            lps.push(clp);
        }
        for _ in 0..no {
            lps.push(olp);
        }
        let log_z = logsumexp(&lps[..]);
        (vlp - log_z, clp - log_z, olp - log_z, ilp - log_z)
    }
    fn logprior_term(
        &self,
        term: &Term,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        let proposed_type = schema.instantiate(&mut self.ctx.write().expect("poisoned context"));
        let actual_type = self
            .infer_term(term, &mut HashMap::new())?
            .instantiate_owned(&mut self.ctx.write().expect("poisoned context"));
        if self.unify(&proposed_type, &actual_type).is_err() {
            Ok(NEG_INFINITY)
        } else {
            self.logprior_term_internal(term, &proposed_type, atom_weights, invent, &mut vec![])
        }
    }
    fn logprior_term_internal(
        &self,
        term: &Term,
        tp: &Type,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        variables: &mut Vec<Variable>,
    ) -> Result<f64, SampleError> {
        let tp = tp.apply_compress(&mut self.ctx.write().expect("poisoned context"));
        // setup the existing options
        let (vs, cs, os) = self.prepare_prior_options(&tp, term, invent, variables);
        // compute the log probability of each kind of head
        let (vlp, clp, olp, ilp) =
            Lex::atom_priors(invent, atom_weights, (vs.len(), cs.len(), os.len()));
        // find the selected option
        let mut options = vs.into_iter().chain(cs).chain(os);
        let option = options
            .find(|&(ref o, _)| o == &Some(term.head()))
            .and_then(|(o, _)| {
                let atom = o.unwrap();
                // use this propagate the type constraints forward
                let arg_types = self.fit_atom(&atom, &tp, false).ok()?;
                Some((atom, arg_types))
            });
        // compute the probability of the term
        match option {
            None => Ok(NEG_INFINITY),
            Some((Atom::Variable(ref v), _)) if !variables.contains(v) => {
                variables.push(v.clone());
                Ok(ilp)
            }
            Some((Atom::Variable(_), _)) => Ok(vlp),
            Some((Atom::Operator(_), ref arg_types)) if arg_types.is_empty() => Ok(clp),
            Some((Atom::Operator(_), arg_types)) => {
                let mut lp = olp;
                for (subterm, mut arg_tp) in term.args().iter().zip(arg_types) {
                    arg_tp.apply_mut_compress(&mut self.ctx.write().expect("poisoned context"));
                    lp += self.logprior_term_internal(
                        subterm,
                        &arg_tp,
                        atom_weights,
                        invent,
                        variables,
                    )?;
                    if lp == NEG_INFINITY {
                        return Ok(lp);
                    }
                }
                Ok(lp)
            }
        }
    }
    fn logprior_rule(
        &self,
        rule: &Rule,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        let schema = self.infer_rule(rule, &mut HashMap::new())?;
        let tp = schema.instantiate(&mut self.ctx.write().expect("poisoned context"));
        let mut variables = vec![];
        let lp_lhs =
            self.logprior_term_internal(&rule.lhs, &tp, atom_weights, invent, &mut variables)?;
        let mut lp = 0.0;
        for rhs in &rule.rhs {
            lp += lp_lhs
                + self.logprior_term_internal(&rhs, &tp, atom_weights, false, &mut variables)?;
        }
        Ok(lp)
    }
    fn logprior_string_rule(
        &self,
        rule: &Rule,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
    ) -> Result<f64, SampleError> {
        let schema = self.infer_rule(rule, &mut HashMap::new())?;
        let tp = schema.instantiate(&mut self.ctx.write().expect("poisoned context"));
        let lp_lhs =
            self.logprior_term_internal(&rule.lhs, &tp, atom_weights, invent, &mut vec![])?;
        let mut lp = 0.0;
        for rhs in &rule.rhs {
            lp += lp_lhs
                + UntypedTRS::p_string(&rule.lhs, rhs, dist, t_max, d_max, &self.signature)
                    .unwrap_or(NEG_INFINITY);
        }
        Ok(lp)
    }
    fn logprior_utrs(
        &self,
        utrs: &UntypedTRS,
        p_number_of_clauses: Box<dyn Fn(usize) -> f64>,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        let clauses = utrs.clauses();
        let mut p_clauses = 0.0;
        for clause in clauses.iter() {
            p_clauses += self.logprior_rule(clause, atom_weights, invent)?;
        }
        Ok(p_number_of_clauses(clauses.len()) + p_clauses)
    }
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::too_many_arguments))]
    fn logprior_srs(
        &self,
        utrs: &UntypedTRS,
        p_number_of_clauses: Box<dyn Fn(usize) -> f64>,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
    ) -> Result<f64, SampleError> {
        let clauses = utrs.clauses();
        let mut p_clauses = 0.0;
        for clause in clauses.iter() {
            p_clauses +=
                self.logprior_string_rule(clause, atom_weights, invent, dist, t_max, d_max)?;
        }
        Ok(p_number_of_clauses(clauses.len()) + p_clauses)
    }
    fn enumerate_n_terms_internal(
        &mut self,
        schema: &TypeSchema,
        invent: bool,
        associative: bool,
        n: usize,
        variables: &mut Vec<Variable>,
    ) -> Vec<Term> {
        self.prepare_options(schema, invent, true, variables)
            .into_iter()
            .flat_map(|atom| {
                self.enumerate_n_terms_from_context(
                    Context::from(atom),
                    invent,
                    associative,
                    n,
                    variables,
                )
            })
            .collect_vec()
    }
    fn enumerate_n_terms_from_context(
        &mut self,
        context: Context,
        invent: bool,
        associative: bool,
        n: usize,
        variables: &mut Vec<Variable>,
    ) -> Vec<Term> {
        match n.cmp(&context.size()) {
            Ordering::Less => vec![],
            Ordering::Equal if context.holes().is_empty() => context
                .to_term()
                .map(|t| vec![t])
                .unwrap_or_else(|()| vec![]),
            _ => {
                let holes = context.holes();
                if holes.is_empty() {
                    vec![]
                } else {
                    // typecheck and get the type of the first hole
                    let hole = &holes[0];
                    let complex = !associative || hole.iter().max() == Some(&0);
                    let mut tps = HashMap::new();
                    if self
                        .infer_context_internal(&context, vec![], &mut tps)
                        .is_err()
                    {
                        return vec![];
                    }
                    let lex_vars = self.free_vars_applied();
                    let schema = tps[hole]
                        .apply_compress(&mut self.ctx.write().expect("poisoned context"))
                        .generalize(&lex_vars);
                    // find all the atoms that could fill that hole; recursively try each
                    self.prepare_options(&schema, invent, complex, variables)
                        .into_iter()
                        .map(Context::from)
                        .flat_map(|subcontext| {
                            let mut new_vars = variables.clone();
                            if let Context::Variable(ref v) = subcontext {
                                if !new_vars.contains(v) {
                                    new_vars.push(v.clone());
                                }
                            }
                            let new_context = context.replace(hole, subcontext).unwrap();
                            self.enumerate_n_terms_from_context(
                                new_context,
                                invent,
                                associative,
                                n,
                                &mut new_vars,
                            )
                        })
                        .collect_vec()
                }
            }
        }
    }
}
impl<'a, 'b, 'c> From<&'a GPLexicon<'b>> for &'a Lexicon<'b> {
    fn from(gp_lex: &'a GPLexicon<'b>) -> &'a Lexicon<'b> {
        &gp_lex.lexicon
    }
}

type Parents<'a, 'b> = Vec<TRS<'a, 'b>>;
type Tried<'a, 'b> = HashMap<TRSMoveName, Vec<Parents<'a, 'b>>>;
pub struct GPLexicon<'a> {
    pub lexicon: Lexicon<'a>,
    pub bg: &'a [Rule],
    pub contexts: Vec<RuleContext>,
    pub(crate) tried: Arc<RwLock<Tried<'a, 'a>>>,
}
impl<'a> GPLexicon<'a> {
    pub fn new<'b>(lex: &Lexicon<'b>, bg: &'b [Rule], contexts: Vec<RuleContext>) -> GPLexicon<'b> {
        let lexicon = lex.clone();
        let tried = Arc::new(RwLock::new(HashMap::new()));
        GPLexicon {
            lexicon,
            bg,
            tried,
            contexts,
        }
    }
    pub fn clear(&self) {
        let mut tried = self.tried.write().expect("poisoned");
        *tried = HashMap::new();
    }
    pub fn add(&self, name: TRSMoveName, parents: Parents<'a, 'a>) {
        let mut tried = self.tried.write().expect("poisoned");
        let entry = tried.entry(name).or_insert_with(|| vec![]);
        entry.push(parents);
    }
    pub fn check(&self, name: TRSMoveName, parents: Parents<'a, 'a>) -> Option<Parents<'a, 'a>> {
        let mut tried = self.tried.write().expect("poisoned");
        let past_parents = tried.entry(name).or_insert_with(|| vec![]);
        if self.novelty_possible(name, &parents, &past_parents) {
            Some(parents)
        } else {
            None
        }
    }
    fn novelty_possible(&self, name: TRSMoveName, parents: &[TRS], past: &[Parents]) -> bool {
        let check = || !past.iter().any(|p| &p[..] == parents);
        match name {
            TRSMoveName::Memorize => past.is_empty(),
            TRSMoveName::SampleRule | TRSMoveName::RegenerateRule => true,
            // | TRSMoveName::Recurse
            // | TRSMoveName::RecurseVariablize
            // | TRSMoveName::RecurseGeneralize => true,
            TRSMoveName::LocalDifference => parents[0].num_learned_rules() > 1 || check(),
            _ => check(),
        }
    }
}
impl<'a> GP for GPLexicon<'a> {
    type Representation = Lexicon<'a>;
    type Expression = TRS<'a, 'a>;
    type Params = GeneticParams;
    type Observation = Vec<Rule>;
    fn genesis<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        pop_size: usize,
        _tp: &TypeSchema,
    ) -> Vec<Self::Expression> {
        match TRS::new(&self.lexicon, params.deterministic, self.bg, vec![]) {
            Ok(mut trs) => {
                if params.deterministic {
                    trs.utrs.make_deterministic();
                }
                let mut pop = Vec::with_capacity(pop_size);
                while pop.len() < pop_size {
                    if let Ok(mut new_trs) = trs.sample_rule(
                        &self.contexts,
                        params.atom_weights,
                        params.max_sample_size,
                        rng,
                    ) {
                        if new_trs[0].unique_shape(&pop) {
                            pop.append(&mut new_trs);
                        }
                    }
                }
                pop
            }
            Err(err) => panic!("invalid background knowledge: {}", err),
        }
    }
    fn reproduce<R: Rng>(
        &self,
        rng: &mut R,
        params: &Self::Params,
        obs: &Self::Observation,
        tournament: &Tournament<Self::Expression>,
    ) -> Vec<Self::Expression> {
        let weights = params.moves.iter().map(|mv| mv.weight).collect_vec();
        let dist = WeightedIndex::new(weights).unwrap();
        loop {
            // Choose a move
            let choice = dist.sample(rng);
            let mv = params.moves[choice].mv;
            let name = mv.name();
            // Sample the parents.
            let parents = mv.get_parents(&tournament, rng);
            // Check the parents.
            if let Some(parents) = self.check(name, parents) {
                // Take the move.
                if let Ok(trss) = mv.take(
                    &self.lexicon,
                    params.deterministic,
                    self.bg,
                    &self.contexts,
                    obs,
                    rng,
                    &parents,
                ) {
                    self.add(name, parents);
                    return trss;
                }
            }
        }
    }
    fn validate_offspring(
        &self,
        _params: &Self::Params,
        population: &[(Self::Expression, f64)],
        _children: &[Self::Expression],
        seen: &mut Vec<Self::Expression>,
        offspring: &mut Vec<Self::Expression>,
        max_validated: usize,
    ) {
        let mut validated = 0;
        while validated < max_validated && validated < offspring.len() {
            let x = &offspring[validated];
            let pop_unique = !population.iter().any(|p| TRS::same_shape(&p.0, &x));
            if pop_unique && x.unique_shape(seen) {
                validated += 1;
            } else {
                offspring.swap_remove(validated);
            }
        }
        offspring.truncate(validated);
        seen.extend_from_slice(&offspring);
    }
}

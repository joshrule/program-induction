use itertools::Itertools;
use polytype::{Context as TypeContext, Type, TypeSchema, Variable as TypeVar};
use rand::{distributions::Distribution, distributions::WeightedIndex, Rng};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::f64::NEG_INFINITY;
use std::fmt;
use std::iter;
use std::sync::{Arc, RwLock};
use term_rewriting::{
    Atom, Context, Operator, PStringDist, Place, Rule, RuleContext, Signature, Term, Variable,
    TRS as UntypedTRS,
};

use super::{SampleError, TypeError, TRS};
use utils::{logsumexp, weighted_sample};
use GP;

type LOpt = (Option<Atom>, Vec<Type>);

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
/// Parameters for [`Lexicon`] genetic programming ([`GP`]).
///
/// [`Lexicon`]: struct.Lexicon.html
/// [`GP`]: ../trait.GP.html
pub struct GeneticParams {
    /// The number of hypotheses crossover should generate.
    pub n_crosses: usize,
    /// The maximum number of nodes a sampled `Term` can have without failing.
    pub max_sample_size: usize,
    /// The probability of keeping a rule during crossover.
    pub p_keep: f64,
    /// The weight to assign variables, constants, and non-constant operators, respectively.
    pub atom_weights: (f64, f64, f64, f64),
}

pub struct ContextPoint<O, E> {
    result: Result<O, E>,
    snapshot: usize,
    lex: Lexicon,
}
impl<O, E> ContextPoint<O, E> {
    pub fn drop(self) -> Result<O, E> {
        let mut lex = self.lex.0.write().expect("poisoned lexicon");
        lex.ctx.rollback(self.snapshot);
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
pub struct Lexicon(pub(crate) Arc<RwLock<Lex>>);
impl Lexicon {
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
    /// # fn main() {
    /// let operators = vec![
    ///     (2, Some("PLUS".to_string()), ptp![@arrow[tp!(int), tp!(int), tp!(int)]]),
    ///     (1, Some("SUCC".to_string()), ptp![@arrow[tp!(int), tp!(int)]]),
    ///     (0, Some("ZERO".to_string()), ptp![int]),
    /// ];
    /// let deterministic = false;
    ///
    /// let lexicon = Lexicon::new(operators, deterministic, TypeContext::default());
    /// # }
    /// ```
    ///
    /// [`polytype::ptp`]: https://docs.rs/polytype/~6.0/polytype/macro.ptp.html
    /// [`polytype::TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`term_rewriting::Operator`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Operator.html
    pub fn new(
        operators: Vec<(u32, Option<String>, TypeSchema)>,
        deterministic: bool,
        ctx: TypeContext,
    ) -> Lexicon {
        let mut signature = Signature::default();
        let mut ops = Vec::with_capacity(operators.len());
        for (id, name, tp) in operators {
            signature.new_op(id, name);
            ops.push(tp);
        }
        Lexicon(Arc::new(RwLock::new(Lex {
            ops,
            vars: vec![],
            signature,
            background: vec![],
            templates: vec![],
            deterministic,
            ctx,
        })))
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
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let vars = vec![];
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some("PLUS".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// sig.new_op(1, Some("SUCC".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    /// sig.new_op(0, Some("ZERO".to_string()));
    /// ops.push(ptp![int]);
    ///
    /// let background = vec![
    ///     parse_rule(&mut sig, "PLUS(x_ ZERO) = x_").expect("parsed rule"),
    ///     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))").expect("parsed rule"),
    /// ];
    ///
    /// let templates = vec![
    ///     parse_rulecontext(&mut sig, "[!] = [!]").expect("parsed rulecontext"),
    /// ];
    ///
    /// let deterministic = false;
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, background, templates, deterministic, TypeContext::default());
    /// # }
    /// ```
    ///
    /// [`polytype::ptp`]: https://docs.rs/polytype/~6.0/polytype/macro.ptp.html
    /// [`polytype::TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`term_rewriting::Signature`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Signature.html
    /// [`term_rewriting::Operator`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Operator.html
    /// [`term_rewriting::Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
    /// [`term_rewriting::Variable`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Variable.html
    /// [`term_rewriting::RuleContext`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.RuleContext.html
    pub fn from_signature(
        signature: Signature,
        ops: Vec<TypeSchema>,
        vars: Vec<TypeSchema>,
        background: Vec<Rule>,
        templates: Vec<RuleContext>,
        deterministic: bool,
        ctx: TypeContext,
    ) -> Lexicon {
        Lexicon(Arc::new(RwLock::new(Lex {
            ops,
            vars,
            signature,
            background,
            templates,
            deterministic,
            ctx,
        })))
    }
    /// Return the specified operator if possible.
    pub fn has_op(&self, name: Option<&str>, arity: u32) -> Result<Operator, ()> {
        let sig = &self.0.read().expect("poisoned lexicon").signature;
        sig.operators()
            .into_iter()
            .find(|op| {
                op.arity() == arity && op.name().as_ref().map(std::string::String::as_str) == name
            })
            .ok_or(())
    }
    /// All the [`term_rewriting::Variable`]s in the `Lexicon`.
    ///
    /// [`term_rewriting::Variable`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Variable.html
    pub fn variables(&self) -> Vec<Variable> {
        self.0.read().expect("poisoned lexicon").variables()
    }
    /// All the free type variables in the `Lexicon`.
    pub fn free_vars(&self) -> Vec<TypeVar> {
        self.0.read().expect("poisoned lexicon").free_vars()
    }
    /// Add a new variable to the `Lexicon`.
    pub fn invent_variable(&mut self, tp: &Type) -> Variable {
        self.0
            .write()
            .expect("poisoned lexicon")
            .invent_variable(tp)
    }
    /// Return the `Lexicon`'s [`TypeContext`].
    ///
    /// [`TypeContext`]: https://docs.rs/polytype/~6.0/polytype/struct.Context.html
    pub fn context(&self) -> TypeContext {
        self.0.read().expect("poisoned lexicon").ctx.clone()
    }
    pub fn snapshot(&self) -> usize {
        self.0.read().expect("poisoned lexicon").ctx.len()
    }
    pub fn rollback(&self, snapshot: usize) {
        self.0
            .write()
            .expect("poisoned lexicon")
            .ctx
            .rollback(snapshot);
    }
    pub fn instantiate(&self, schema: &TypeSchema) -> Type {
        schema.instantiate(&mut self.0.write().expect("poisoned lexicon").ctx)
    }
    pub fn unify(&self, t1: &Type, t2: &Type) -> Result<(), ()> {
        self.0
            .write()
            .expect("poisoned lexicon")
            .ctx
            .unify(t1, t2)
            .map_err(|_| ())
    }
    /// Return a new [`Type::Variable`] from the `Lexicon`'s [`TypeContext`].
    ///
    /// [`Type::Variable`]: https://docs.rs/polytype/~6.0/polytype/enum.Type.html
    /// [`TypeContext`]: https://docs.rs/polytype/~6.0/polytype/struct.Context.html
    pub fn fresh_type_variable(&self) -> Type {
        self.0.write().expect("poisoned lexicon").ctx.new_variable()
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
    /// # fn main() {
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
    /// let background = vec![];
    /// let templates = vec![];
    ///
    /// let deterministic = false;
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, background, templates, deterministic, TypeContext::default());
    ///
    /// let context = Context::Application {
    ///     op: succ,
    ///     args: vec![Context::Hole]
    /// };
    /// let mut ctx = lexicon.context();
    ///
    /// let inferred_schema = lexicon.infer_context(&context, &mut ctx).unwrap();
    ///
    /// assert_eq!(inferred_schema, ptp![int]);
    /// # }
    /// ```
    ///
    /// [`polytype::TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`term_rewriting::Context`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Context.html
    pub fn infer_context(&self, context: &Context) -> ContextPoint<TypeSchema, TypeError> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.infer_context(context);
        ContextPoint {
            snapshot,
            result,
            lex: self.clone(),
        }
    }
    /// Infer the `TypeSchema` associated with a `RuleContext`.
    pub fn infer_rulecontext(&self, context: &RuleContext) -> Result<TypeSchema, TypeError> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.infer_rulecontext(context);
        lex.ctx.rollback(snapshot);
        result
    }
    /// Infer the `TypeSchema` associated with a `Rule`.
    pub fn infer_rule(
        &self,
        rule: &Rule,
        types: &mut HashMap<Place, Type>,
    ) -> Result<TypeSchema, TypeError> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.infer_rule(rule, types);
        for (_, v) in types.iter_mut() {
            v.apply_mut(&lex.ctx);
            v.apply_mut(&lex.ctx);
        }
        lex.ctx.rollback(snapshot);
        result
    }
    /// Infer the `TypeSchema` associated with a collection of `Rules`.
    pub fn infer_rules(&self, rules: &[Rule]) -> Result<TypeSchema, TypeError> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.infer_rules(rules);
        lex.ctx.rollback(snapshot);
        result
    }
    /// Infer the `TypeSchema` associated with a subterm in a `Rule`.
    pub fn infer_subrule(
        &self,
        rule: &Rule,
        subterm: &Term,
    ) -> ContextPoint<TypeSchema, TypeError> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.infer_subrule(rule, subterm);
        ContextPoint {
            snapshot,
            result,
            lex: self.clone(),
        }
    }
    /// Infer the `TypeSchema` associated with a `Rule`.
    pub fn infer_op(&self, op: &Operator) -> Result<TypeSchema, TypeError> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.op_tp(op.clone()).map(|tp| tp.clone());
        lex.ctx.rollback(snapshot);
        result
    }
    pub fn infer_term(&self, term: &Term) -> ContextPoint<TypeSchema, TypeError> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.infer_term(term);
        ContextPoint {
            snapshot,
            result,
            lex: self.clone(),
        }
    }
    pub fn infer_utrs(&self, utrs: &UntypedTRS) -> Result<(), TypeError> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.infer_utrs(utrs);
        lex.ctx.rollback(snapshot);
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
    /// # fn main() {
    /// let operators = vec![
    ///     (2, Some("PLUS".to_string()), ptp![@arrow[tp!(int), tp!(int), tp!(int)]]),
    ///     (1, Some("SUCC".to_string()), ptp![@arrow[tp!(int), tp!(int)]]),
    ///     (0, Some("ZERO".to_string()), ptp![int]),
    /// ];
    /// let deterministic = false;
    /// let mut lexicon = Lexicon::new(operators, deterministic, TypeContext::default());
    ///
    /// let schema = ptp![int];
    /// let mut ctx = lexicon.context();
    /// let invent = true;
    /// let variable = true;
    /// let atom_weights = (0.5, 0.25, 0.25);
    /// let max_size = 50;
    ///
    /// let term = lexicon.sample_term(&schema, &mut ctx, atom_weights, invent, variable, max_size).unwrap();
    /// # }
    /// ```
    ///
    /// [`term_rewriting::Term`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Term.html
    pub fn sample_term(
        &self,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        variable: bool,
        max_size: usize,
        vars: &mut Vec<Variable>,
    ) -> Result<Term, SampleError> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.sample_term(schema, atom_weights, invent, variable, max_size, vars);
        lex.ctx.rollback(snapshot);
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
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result =
            lex.sample_term_from_context(context, atom_weights, invent, variable, max_size);
        lex.ctx.rollback(snapshot);
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
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.sample_rule(schema, atom_weights, invent, max_size);
        lex.ctx.rollback(snapshot);
        result
    }
    /// Sample a `Rule` conditioned on a `Context` rather than a `TypeSchema`.
    pub fn sample_rule_from_context(
        &self,
        context: RuleContext,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        max_size: usize,
    ) -> ContextPoint<Rule, SampleError> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.sample_rule_from_context(context, atom_weights, invent, max_size);
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
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let vec = lex.enumerate_to_n_terms(schema, invent, associative, n);
        lex.ctx.rollback(snapshot);
        vec
    }
    pub fn enumerate_n_terms(
        &mut self,
        schema: &TypeSchema,
        invent: bool,
        associative: bool,
        n: usize,
    ) -> Vec<Term> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let vec = lex.enumerate_n_terms(schema, invent, associative, n);
        lex.ctx.rollback(snapshot);
        vec
    }
    pub fn enumerate_to_n_rules(
        &mut self,
        schema: &TypeSchema,
        invent: bool,
        associative: bool,
        n: usize,
    ) -> Vec<Rule> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let vec = lex.enumerate_to_n_rules(schema, invent, associative, n);
        lex.ctx.rollback(snapshot);
        vec
    }
    pub fn enumerate_n_rules(
        &mut self,
        schema: &TypeSchema,
        invent: bool,
        associative: bool,
        n: usize,
    ) -> Vec<Rule> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let vec = lex.enumerate_n_rules(schema, invent, associative, n);
        lex.ctx.rollback(snapshot);
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
        let mut lex = self.0.write().expect("posioned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.logprior_term(term, schema, atom_weights, invent);
        lex.ctx.rollback(snapshot);
        result
    }
    /// Give the log probability of sampling a Rule.
    pub fn logprior_rule(
        &self,
        rule: &Rule,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.logprior_rule(rule, atom_weights, invent);
        lex.ctx.rollback(snapshot);
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
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.logprior_utrs(utrs, p_number_of_rules, atom_weights, invent);
        lex.ctx.rollback(snapshot);
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
        let mut lex = self.0.write().expect("poisoned lexicon");
        let snapshot = lex.ctx.len();
        let result = lex.logprior_srs(
            utrs,
            p_number_of_rules,
            atom_weights,
            invent,
            dist,
            t_max,
            d_max,
        );
        lex.ctx.rollback(snapshot);
        result
    }
    /// merge two `TRS` into a single `TRS`.
    pub fn combine(&self, trs1: &TRS, trs2: &TRS) -> Result<TRS, TypeError> {
        assert_eq!(trs1.lex, trs2.lex);
        let background_size = trs1
            .lex
            .0
            .read()
            .expect("poisoned lexicon")
            .background
            .len();
        let rules1 = trs1.utrs.rules[..(trs1.utrs.len() - background_size)].to_vec();
        let rules2 = trs2.utrs.rules[..(trs2.utrs.len() - background_size)].to_vec();
        let mut trs = TRS::new(&trs1.lex, rules1)?;
        let filtered_rules = rules2
            .into_iter()
            .flat_map(|r| r.clauses())
            .filter(|r2| {
                for r1 in &trs.utrs().clauses() {
                    if Rule::alpha(r1, r2).is_some()
                        || (self.0.read().expect("poisoned lexicon").deterministic
                            && Term::alpha(&r1.lhs, &r2.lhs).is_some())
                    {
                        return false;
                    }
                }
                true
            })
            .collect_vec();
        trs.utrs.pushes(filtered_rules).unwrap(); // hack?
        if self.0.read().expect("poisoned lexicon").deterministic {
            trs.utrs.make_deterministic();
        }
        Ok(trs)
    }
}
impl fmt::Debug for Lexicon {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let lex = self.0.read();
        write!(f, "Lexicon({:?})", lex)
    }
}
impl PartialEq for Lexicon {
    fn eq(&self, other: &Lexicon) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl fmt::Display for Lexicon {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.read().expect("poisoned lexicon").fmt(f)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Lex {
    pub(crate) ops: Vec<TypeSchema>,
    pub(crate) vars: Vec<TypeSchema>,
    pub(crate) signature: Signature,
    pub(crate) background: Vec<Rule>,
    /// Rule templates to use when sampling rules.
    pub(crate) templates: Vec<RuleContext>,
    /// If `true`, then the `TRS`s should be deterministic.
    pub(crate) deterministic: bool,
    pub(crate) ctx: TypeContext,
}
impl fmt::Display for Lex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Signature:")?;
        for (op, schema) in self.signature.operators().iter().zip(&self.ops) {
            writeln!(f, "{}: {}", op.display(), schema)?;
        }
        for (var, schema) in self.signature.variables().iter().zip(&self.vars) {
            writeln!(f, "{}: {}", var.display(), schema)?;
        }
        writeln!(f, "\nBackground: {}", self.background.len())?;
        for rule in &self.background {
            writeln!(f, "{}", rule.pretty())?;
        }
        writeln!(f, "\nTemplates: {}", self.templates.len())?;
        for template in &self.templates {
            writeln!(f, "{}", template.pretty())?;
        }
        writeln!(f, "\nDeterministic: {}", self.deterministic)
    }
}
impl Lex {
    fn variables(&self) -> Vec<Variable> {
        self.signature.variables()
    }
    fn free_vars(&self) -> Vec<TypeVar> {
        let vars_fvs = self.vars.iter().flat_map(TypeSchema::free_vars);
        let ops_fvs = self.ops.iter().flat_map(TypeSchema::free_vars);
        vars_fvs.chain(ops_fvs).unique().collect()
    }
    fn free_vars_applied(&self) -> Vec<TypeVar> {
        self.free_vars()
            .into_iter()
            .flat_map(|x| Type::Variable(x).apply(&self.ctx).vars())
            .unique()
            .collect::<Vec<_>>()
    }
    fn invent_variable(&mut self, tp: &Type) -> Variable {
        let var = self.signature.new_var(None);
        self.vars.push(TypeSchema::Monotype(tp.clone()));
        var
    }
    fn fit_atom(
        &mut self,
        atom: &Atom,
        tp: &Type,
        rollback: bool,
    ) -> Result<Vec<Type>, SampleError> {
        let atom_tp = self.instantiate_atom(atom)?;
        let rollback_n = self.ctx.len();
        let unify_tp = match atom {
            Atom::Operator(o) if o.arity() == 0 => atom_tp.clone(),
            _ => atom_tp
                .returns()
                .map(Type::clone)
                .or_else(|| Some(atom_tp.clone()))
                .unwrap(),
        };
        let result = self.ctx.unify_fast(unify_tp, tp.clone());
        if rollback || result.is_err() {
            self.ctx.rollback(rollback_n);
        }
        result
            .map(|_| match atom {
                Atom::Operator(o) if o.arity() > 0 => atom_tp
                    .args()
                    .map(|o| o.into_iter().cloned().collect())
                    .unwrap_or_else(Vec::new),
                _ => Vec::with_capacity(0),
            })
            .map_err(|e| SampleError::TypeError(TypeError::Unification(e)))
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
            Atom::Variable(ref v) => Ok(Term::Variable(v.clone())),
            Atom::Operator(ref op) => {
                let mut size = 1;
                let snapshot = self.ctx.len(); // for "undo" semantics
                let mut args = Vec::with_capacity(arg_types.len());
                let can_be_variable = true; // subterms can always be variables
                for arg_tp in arg_types {
                    let subtype = arg_tp.apply(&self.ctx);
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
                            let tp = self.infer_term(&subterm)?.instantiate_owned(&mut self.ctx);
                            self.ctx.unify_fast(subtype, tp)?;
                            Ok(subterm)
                        });
                    match result {
                        Ok(subterm) => {
                            size += subterm.size();
                            args.push(subterm);
                        }
                        Err(e) => {
                            self.ctx.rollback(snapshot);
                            return Err(e);
                        }
                    }
                }
                Ok(Term::Application {
                    op: op.clone(),
                    args,
                })
            }
        }
    }
    fn instantiate_atom(&mut self, atom: &Atom) -> Result<Type, TypeError> {
        let schema = self.infer_atom(atom)?.clone();
        let mut tp = schema.instantiate_owned(&mut self.ctx);
        tp.apply_mut(&self.ctx);
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
    fn infer_term(&mut self, term: &Term) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_term_internal(term, vec![], &mut HashMap::new())?;
        let lex_vars = self.free_vars_applied();
        Ok(tp.apply(&self.ctx).generalize(&lex_vars))
    }
    fn infer_term_internal(
        &mut self,
        term: &Term,
        place: Place,
        tps: &mut HashMap<Place, Type>,
    ) -> Result<Type, TypeError> {
        let tp = match *term {
            Term::Variable(ref v) => self.instantiate_atom(&Atom::from(v.clone()))?,
            Term::Application { ref op, ref args } => {
                let head_type = self.instantiate_atom(&Atom::from(op.clone()))?;
                let body_type = {
                    let mut pre_types = Vec::with_capacity(args.len() + 1);
                    for (i, a) in args.iter().enumerate() {
                        let mut new_place = place.clone();
                        new_place.push(i);
                        pre_types.push(self.infer_term_internal(a, new_place, tps)?);
                    }
                    pre_types.push(self.ctx.new_variable());
                    Type::from(pre_types)
                };
                self.ctx.unify(&head_type, &body_type)?;
                if op.arity() > 0 {
                    head_type.returns().unwrap_or(&head_type).apply(&self.ctx)
                } else {
                    head_type.apply(&self.ctx)
                }
            }
        };
        tps.insert(place, tp.clone());
        Ok(tp)
    }
    fn infer_context(&mut self, context: &Context) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_context_internal(context, vec![], &mut HashMap::new())?;
        let lex_vars = self.free_vars_applied();
        Ok(tp.apply(&self.ctx).generalize(&lex_vars))
    }
    fn infer_context_internal(
        &mut self,
        context: &Context,
        place: Place,
        tps: &mut HashMap<Place, Type>,
    ) -> Result<Type, TypeError> {
        let tp = match *context {
            Context::Hole => self.ctx.new_variable(),
            Context::Variable(ref v) => self.instantiate_atom(&Atom::from(v.clone()))?,
            Context::Application { ref op, ref args } => {
                let head_type = self.instantiate_atom(&Atom::from(op.clone()))?;
                let body_type = {
                    let mut pre_types = Vec::with_capacity(args.len() + 1);
                    for (i, a) in args.iter().enumerate() {
                        let mut new_place = place.clone();
                        new_place.push(i);
                        pre_types.push(self.infer_context_internal(a, new_place, tps)?);
                    }
                    pre_types.push(self.ctx.new_variable());
                    Type::from(pre_types)
                };
                self.ctx.unify(&head_type, &body_type)?;
                if op.arity() == 0 {
                    head_type.apply(&self.ctx)
                } else {
                    head_type.returns().unwrap_or(&head_type).apply(&self.ctx)
                }
            }
        };
        tps.insert(place, tp.clone());
        Ok(tp)
    }
    fn infer_rule(
        &mut self,
        rule: &Rule,
        tps: &mut HashMap<Place, Type>,
    ) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_rule_internal(rule, tps)?;
        let lex_vars = self.free_vars_applied();
        Ok(tp.apply(&self.ctx).generalize(&lex_vars))
    }
    fn infer_rule_internal(
        &mut self,
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
            self.ctx.unify(&lhs_type, &rhs_type)?;
        }
        Ok(lhs_type.apply(&self.ctx))
    }
    fn infer_rules(&mut self, rules: &[Rule]) -> Result<TypeSchema, TypeError> {
        let rule_tps = rules
            .iter()
            .map(|rule| {
                self.infer_rule(rule, &mut HashMap::new())
                    .map(|rule_tp| rule_tp.instantiate(&mut self.ctx))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tp = self.ctx.new_variable();
        for rule_tp in rule_tps {
            self.ctx.unify(&tp, &rule_tp)?;
        }
        let lex_vars = self.free_vars_applied();
        Ok(tp.apply(&self.ctx).generalize(&lex_vars))
    }
    fn infer_subrule(&mut self, rule: &Rule, subterm: &Term) -> Result<TypeSchema, TypeError> {
        // infer the rule
        let mut tps = HashMap::new();
        self.infer_rule(&rule, &mut tps)?;

        // find the places where that term occurs in the rules
        let places = rule
            .subterms()
            .into_iter()
            .filter_map(|(x, p)| if x == subterm { Some(p) } else { None })
            .collect_vec();

        // unify all of these
        let tp = self.ctx.new_variable();
        for place in &places {
            self.ctx.unify(&tp, &tps[place])?;
        }

        // compute the final type
        let lex_vars = self.free_vars_applied();
        if places.is_empty() {
            Err(TypeError::Malformed)
        } else {
            Ok(tp.apply(&self.ctx).generalize(&lex_vars))
        }
    }
    fn infer_rulecontext(&mut self, context: &RuleContext) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_rulecontext_internal(context, &mut HashMap::new())?;
        let lex_vars = self.free_vars_applied();
        Ok(tp.apply(&self.ctx).generalize(&lex_vars))
    }
    fn infer_rulecontext_internal(
        &mut self,
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
            self.ctx.unify(&lhs_type, &rhs_type)?;
        }
        Ok(lhs_type.apply(&self.ctx))
    }
    fn infer_utrs(&mut self, utrs: &UntypedTRS) -> Result<(), TypeError> {
        // TODO: we assume the variables already exist in the signature. Is that sensible?
        for rule in &utrs.rules {
            self.infer_rule(rule, &mut HashMap::new())?;
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
        let tp = schema.instantiate(&mut self.ctx);
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
        let option = weighted_sample(&options, &weights).clone();
        let atom = option.unwrap_or_else(|| {
            let new_var = self.invent_variable(tp);
            vars.push(new_var.clone());
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
            let schema = &map[p].apply(&self.ctx).generalize(&lex_vars);
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
            let snapshot = self.ctx.len();
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
                self.ctx.rollback(snapshot);
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
            let schema = TypeSchema::Monotype(tps[p].apply(&self.ctx));
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
        let snapshot = self.ctx.len();
        let lhss = self.enumerate_to_n_terms(schema, invent, associative, n - 1);
        self.ctx.rollback(snapshot);
        lhss.iter()
            .flat_map(|lhs| {
                let snapshot = self.ctx.len();
                let rhss = if let Ok(schema) = self.infer_term(lhs) {
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
                self.ctx.rollback(snapshot);
                rhss.into_iter()
                    .filter_map(|rhs| {
                        Rule::new(lhs.clone(), vec![rhs]).and_then(|rule| {
                            let snapshot = self.ctx.len();
                            let result = self.infer_rule(&rule, &mut HashMap::new());
                            self.ctx.rollback(snapshot);
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
        let tp = schema.instantiate(&mut self.ctx);
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
        &mut self,
        tp: &Type,
        term: &Term,
        invent: bool,
        vars: &[Variable],
    ) -> (Vec<LOpt>, Vec<LOpt>, Vec<LOpt>) {
        let mut vs = vars
            .iter()
            .cloned()
            .filter_map(|v| {
                let atom = Atom::Variable(v.clone());
                if let Ok(arg_types) = self.fit_atom(&atom, &tp, true) {
                    Some((Some(Atom::Variable(v.clone())), arg_types))
                } else {
                    None
                }
            })
            .collect_vec();
        if invent {
            // empty arg_types below because variables have no arguments
            if let Term::Variable(ref v) = *term {
                if !vars.contains(v) {
                    vs.push((Some(Atom::Variable(v.clone())), vec![]));
                } else {
                    vs.push((None, vec![]));
                }
            } else {
                vs.push((None, vec![]));
            }
        }
        let (mut cs, mut os) = (vec![], vec![]);
        for op in &self.signature.operators() {
            let atom = Atom::Operator(op.clone());
            if let Ok(arg_types) = self.fit_atom(&atom, &tp, true) {
                if op.arity() == 0 {
                    // arg_types is empty, but using it avoids allocation
                    cs.push((Some(atom.clone()), arg_types));
                } else {
                    os.push((Some(atom.clone()), arg_types));
                }
            }
        }
        (vs, cs, os)
    }
    fn atom_priors(
        invent: bool,
        (vw, cw, ow, iw): (f64, f64, f64, f64),
        (vs, cs, os): (&[LOpt], &[LOpt], &[LOpt]),
    ) -> (f64, f64, f64, f64) {
        let z = vw + cw + ow;
        let (vp, cp, op) = (vw / z, cw / z, ow / z);
        let vlp = if invent && vs.len() == 1 {
            // in this case, the only variable is invented, so no mass goes here
            NEG_INFINITY
        } else {
            vp.ln() - ((vs.len() as f64) + (invent as usize as f64) * (-1.0 + iw)).ln()
        };
        let ilp = if invent {
            vp.ln() + iw.ln() - ((vs.len() as f64) - 1.0 + iw).ln()
        } else {
            NEG_INFINITY
        };
        let clp = if cs.is_empty() {
            NEG_INFINITY
        } else {
            cp.ln() - (cs.len() as f64).ln()
        };
        let olp = if os.is_empty() {
            NEG_INFINITY
        } else {
            op.ln() - (os.len() as f64).ln()
        };
        let mut lps = vec![];
        for i in 0..vs.len() {
            if invent && i == 0 {
                lps.push(ilp);
            } else {
                lps.push(vlp);
            }
        }
        for _ in 0..cs.len() {
            lps.push(clp);
        }
        for _ in 0..os.len() {
            lps.push(olp);
        }
        let log_z = logsumexp(&lps[..]);
        (vlp - log_z, clp - log_z, olp - log_z, ilp - log_z)
    }
    fn logprior_term(
        &mut self,
        term: &Term,
        schema: &TypeSchema,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        let proposed_type = schema.instantiate(&mut self.ctx);
        let actual_type = self.infer_term(term)?.instantiate_owned(&mut self.ctx);
        if self.ctx.unify(&proposed_type, &actual_type).is_err() {
            Ok(NEG_INFINITY)
        } else {
            self.logprior_term_internal(term, &proposed_type, atom_weights, invent, &mut vec![])
        }
    }
    fn logprior_term_internal(
        &mut self,
        term: &Term,
        tp: &Type,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        variables: &mut Vec<Variable>,
    ) -> Result<f64, SampleError> {
        // setup the existing options
        let (vs, cs, os) = self.prepare_prior_options(tp, term, invent, variables);
        // compute the log probability of each kind of head
        let (vlp, clp, olp, ilp) = Lex::atom_priors(invent, atom_weights, (&vs, &cs, &os));
        // find the selected option
        let mut options = vs.into_iter().chain(cs).chain(os);
        let option_option = options.find(|&(ref o, _)| o == &Some(term.head()));
        let option = option_option.map(|(o, arg_types)| (o.unwrap(), arg_types));
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
                    arg_tp.apply_mut(&self.ctx);
                    lp += self.logprior_term_internal(
                        subterm,
                        &arg_tp,
                        atom_weights,
                        invent,
                        variables,
                    )?;
                }
                Ok(lp)
            }
        }
    }
    fn logprior_rule(
        &mut self,
        rule: &Rule,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        let schema = self.infer_rule(rule, &mut HashMap::new())?;
        let tp = schema.instantiate(&mut self.ctx);
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
        &mut self,
        rule: &Rule,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
    ) -> Result<f64, SampleError> {
        let schema = self.infer_rule(rule, &mut HashMap::new())?;
        let tp = schema.instantiate(&mut self.ctx);
        let lp_lhs =
            self.logprior_term_internal(&rule.lhs, &tp, atom_weights, invent, &mut vec![])?;
        let mut lp = 0.0;
        for rhs in &rule.rhs {
            lp += lp_lhs
                + UntypedTRS::p_string(&rule.lhs, rhs, dist, t_max, d_max).unwrap_or(NEG_INFINITY);
        }
        Ok(lp)
    }
    fn n_learned_clauses(&self, utrs: &UntypedTRS) -> usize {
        let n_clauses: usize = utrs.clauses().len();
        let n_background_clauses: usize = self.background.iter().map(|x| x.clauses().len()).sum();
        n_clauses - n_background_clauses
    }
    fn logprior_utrs(
        &mut self,
        utrs: &UntypedTRS,
        p_number_of_clauses: Box<dyn Fn(usize) -> f64>,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        let n_learned_clauses = self.n_learned_clauses(utrs);
        let mut p_clauses = 0.0;
        for clause in utrs.clauses().iter().take(n_learned_clauses) {
            p_clauses += self.logprior_rule(clause, atom_weights, invent)?;
        }
        Ok(p_number_of_clauses(n_learned_clauses) + p_clauses)
    }
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::too_many_arguments))]
    fn logprior_srs(
        &mut self,
        utrs: &UntypedTRS,
        p_number_of_clauses: Box<dyn Fn(usize) -> f64>,
        atom_weights: (f64, f64, f64, f64),
        invent: bool,
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
    ) -> Result<f64, SampleError> {
        let n_learned_clauses = self.n_learned_clauses(utrs);
        let mut p_clauses = 0.0;
        for clause in utrs.clauses().iter().take(n_learned_clauses) {
            p_clauses +=
                self.logprior_string_rule(clause, atom_weights, invent, dist, t_max, d_max)?;
        }
        Ok(p_number_of_clauses(n_learned_clauses) + p_clauses)
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
                    let schema = tps[hole].apply(&self.ctx).generalize(&lex_vars);
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
impl GP for Lexicon {
    type Expression = TRS;
    type Params = GeneticParams;
    type Observation = Vec<Rule>;
    fn genesis<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        pop_size: usize,
        _tp: &TypeSchema,
    ) -> Vec<Self::Expression> {
        let trs = TRS::new(self, Vec::new());
        match trs {
            Ok(mut trs) => {
                if self.0.read().expect("poisoned lexicon").deterministic {
                    trs.utrs.make_deterministic();
                }
                let templates = self.0.read().expect("poisoned lexicon").templates.clone();
                let mut pop = Vec::with_capacity(pop_size);
                while pop.len() < pop_size {
                    if let Ok(new_trs) = trs.clone().add_rule(
                        &templates,
                        params.atom_weights,
                        params.max_sample_size,
                        rng,
                    ) {
                        if !pop
                            .iter()
                            .any(|p: &TRS| UntypedTRS::alphas(&p.utrs, &new_trs.utrs))
                        {
                            pop.push(new_trs);
                        }
                    }
                }
                pop
            }
            Err(err) => {
                let lex = self.0.read().expect("poisoned lexicon");
                let background_trs = UntypedTRS::new(lex.background.clone());
                panic!(
                    "invalid background knowledge {}: {}",
                    background_trs.display(),
                    err
                )
            }
        }
    }
    fn mutate<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        trs: &Self::Expression,
        obs: &Self::Observation,
    ) -> Vec<Self::Expression> {
        // disallow deleting if you have no rules to delete
        let not_empty = !trs.is_empty() as usize;
        let weights = vec![1, (5 * not_empty), (10 * not_empty), 1, 1, 1, 1];
        let dist = WeightedIndex::new(weights).unwrap();
        loop {
            match dist.sample(rng) {
                // add rule
                0 => {
                    let templates = self.0.read().expect("poisoned lexicon").templates.clone();
                    if let Ok(new_trs) =
                        trs.add_rule(&templates, params.atom_weights, params.max_sample_size, rng)
                    {
                        return vec![new_trs];
                    }
                }
                // replace rule
                1 => {
                    let templates = self.0.read().expect("poisoned lexicon").templates.clone();
                    if let Ok(new_trss) = trs.delete_rule() {
                        return new_trss
                            .into_iter()
                            .map(|trs| {
                                if let Ok(new_trs) = trs.add_rule(
                                    &templates,
                                    params.atom_weights,
                                    params.max_sample_size,
                                    rng,
                                ) {
                                    vec![new_trs]
                                } else {
                                    vec![]
                                }
                            })
                            .flatten()
                            .collect_vec();
                    }
                }
                // delete rule
                2 => {
                    if let Ok(new_trss) = trs.delete_rule() {
                        return new_trss;
                    }
                }
                // regenerate rule
                3 => {
                    if let Ok(new_trss) =
                        trs.regenerate_rule(params.atom_weights, params.max_sample_size, rng)
                    {
                        return new_trss;
                    }
                }
                // add exception
                4 => {
                    if let Ok(new_trss) = trs.add_exception(obs) {
                        return new_trss;
                    }
                }
                // local difference
                5 => {
                    if let Ok(new_trss) = trs.local_difference(rng) {
                        return new_trss;
                    }
                }
                // replace term with variable
                6 => {
                    if let Ok(new_trss) = trs.replace_term_with_var(rng) {
                        return new_trss;
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    fn crossover<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        parent1: &Self::Expression,
        parent2: &Self::Expression,
        _obs: &Self::Observation,
    ) -> Vec<Self::Expression> {
        let trs = self
            .combine(parent1, parent2)
            .expect("poorly-typed TRS in crossover");
        iter::repeat(trs)
            .take(params.n_crosses)
            .update(|trs| {
                trs.utrs.rules.retain(|r| {
                    self.0
                        .read()
                        .expect("poisoned lexicon")
                        .background
                        .contains(&r)
                        || rng.gen_bool(params.p_keep)
                })
            })
            .collect()
    }
    fn abiogenesis<R: Rng>(
        &self,
        _params: &Self::Params,
        _rng: &mut R,
        obs: &Self::Observation,
    ) -> Vec<Self::Expression> {
        if let Ok(trs) = TRS::new(self, obs.clone()) {
            vec![trs]
        } else {
            vec![]
        }
    }
    fn validate_offspring(
        &self,
        _params: &Self::Params,
        population: &[(Self::Expression, f64)],
        children: &[Self::Expression],
        offspring: &mut Vec<Self::Expression>,
    ) {
        // select alpha-unique individuals that are not yet in the population
        offspring.retain(|ref x| {
            (!population
                .iter()
                .any(|p| UntypedTRS::alphas(&p.0.utrs, &x.utrs)))
                & (!children
                    .iter()
                    .any(|c| UntypedTRS::alphas(&c.utrs, &x.utrs)))
        });
        *offspring = offspring.iter().fold(vec![], |mut acc, ref x| {
            if !acc.iter().any(|a| UntypedTRS::alphas(&a.utrs, &x.utrs)) {
                acc.push((*x).clone());
            }
            acc
        });
    }
}

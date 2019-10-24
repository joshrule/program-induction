//! Polymorphically typed (Hindley-Milner) First-Order Term Rewriting Systems (no abstraction)
//!
//! Much thanks to:
//! - https://github.com/rob-smallshire/hindley-milner-python
//! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! - (TAPL; Pierce, 2002, ch. 22)

use itertools::Itertools;
use polytype::{Context as TypeContext, Type};
use rand::{seq::IteratorRandom, seq::SliceRandom, Rng};
use std::collections::HashMap;
use std::f64::NEG_INFINITY;
use std::fmt;
use std::iter::once;
use term_rewriting::{
    trace::Trace, Atom, Context, Operator, Rule, RuleContext, Term, Variable, TRS as UntypedTRS,
};

use super::{Lexicon, Likelihood, ModelParams, Prior, SampleError, TypeError};
use utils::{block_generative_logpdf, fail_geometric_logpdf};

/// Manages the semantics of a term rewriting system.
#[derive(Debug, PartialEq, Clone)]
pub struct TRS {
    pub(crate) lex: Lexicon,
    // INVARIANT: UntypedTRS.rules ends with lex.background
    pub(crate) utrs: UntypedTRS,
}
impl TRS {
    /// Create a new `TRS` under the given [`Lexicon`]. Any background knowledge
    /// will be appended to the given ruleset.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use term_rewriting::{Signature, parse_rule};
    /// # use polytype::Context as TypeContext;
    /// # fn main() {
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
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], vec![], false, TypeContext::default());
    ///
    /// let ctx = lexicon.context();
    ///
    /// let trs = TRS::new(&lexicon, rules).unwrap();
    ///
    /// assert_eq!(trs.size(), 12);
    /// # }
    /// ```
    /// [`Lexicon`]: struct.Lexicon.html
    pub fn new(lexicon: &Lexicon, mut rules: Vec<Rule>) -> Result<TRS, TypeError> {
        let utrs = {
            let utrs = {
                let lex = lexicon.0.read().expect("poisoned lexicon");
                rules.append(&mut lex.background.clone());
                let mut utrs = UntypedTRS::new(rules);
                if lexicon.0.read().expect("poisoned lexicon").deterministic {
                    utrs.make_deterministic();
                }
                utrs
            };
            lexicon.infer_utrs(&utrs)?;
            utrs
        };
        Ok(TRS {
            lex: lexicon.clone(),
            utrs,
        })
    }

    pub fn context(&self) -> TypeContext {
        self.lex.context()
    }

    /// The size of the underlying [`term_rewriting::TRS`].
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn size(&self) -> usize {
        self.utrs.size()
    }

    /// The length of the underlying [`term_rewriting::TRS`].
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn len(&self) -> usize {
        self.utrs.len()
    }

    /// Is the underlying [`term_rewriting::TRS`] empty?.
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn is_empty(&self) -> bool {
        self.utrs.is_empty()
    }

    /// The underlying [`term_rewriting::TRS`].
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn utrs(&self) -> UntypedTRS {
        self.utrs.clone()
    }

    /// A pseudo log prior for a `TRS`: the negative [`size`] of the `TRS`
    /// scaled by some cost per token.
    ///
    /// [`size`]: struct.TRS.html#method.size
    pub fn log_prior(&self, params: ModelParams) -> f64 {
        match params.prior {
            Prior::Size(p_token) => -p_token * (self.size() as f64),
            Prior::SimpleGenerative {
                p_rule,
                atom_weights,
            } => {
                let p_num_rules = Box::new(move |k| fail_geometric_logpdf(k, 1.0 - p_rule));
                self.lex
                    .logprior_utrs(&self.utrs, p_num_rules, atom_weights, true)
                    .unwrap_or(NEG_INFINITY)
            }
            Prior::BlockGenerative {
                p_null,
                p_rule,
                n_blocks,
                atom_weights,
            } => {
                let p_num_rules =
                    Box::new(move |k| block_generative_logpdf(p_null, 1.0 - p_rule, k, n_blocks));
                self.lex
                    .logprior_utrs(&self.utrs, p_num_rules, atom_weights, true)
                    .unwrap_or(NEG_INFINITY)
            }
            Prior::StringBlockGenerative {
                p_null,
                p_rule,
                n_blocks,
                atom_weights,
                dist,
                t_max,
                d_max,
            } => {
                let p_num_rules =
                    Box::new(move |k| block_generative_logpdf(p_null, 1.0 - p_rule, k, n_blocks));
                self.lex
                    .logprior_srs(
                        &self.utrs,
                        p_num_rules,
                        atom_weights,
                        true,
                        dist,
                        t_max,
                        d_max,
                    )
                    .unwrap_or(NEG_INFINITY)
            }
        }
    }

    /// A log likelihood for a `TRS`: the probability of `data`'s RHSs appearing
    /// in [`term_rewriting::Trace`]s rooted at its LHSs.
    ///
    /// [`term_rewriting::Trace`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/trace/struct.Trace.html
    pub fn log_likelihood(&self, data: &[Rule], params: ModelParams) -> f64 {
        data.iter()
            .map(|x| self.single_log_likelihood(x, params))
            .sum()
    }

    /// Compute the log likelihood for a single datum.
    fn single_log_likelihood(&self, datum: &Rule, params: ModelParams) -> f64 {
        if let Some(ref rhs) = datum.rhs() {
            let mut trace = Trace::new(
                &self.utrs,
                &datum.lhs,
                params.p_observe,
                params.max_size,
                params.max_depth,
                params.strategy,
            );
            match params.likelihood {
                // Binary ll: 0 or -\infty
                Likelihood::Binary => {
                    if trace.rewrites_to(params.max_steps, rhs, Box::new(|_, _| 0.0))
                        == NEG_INFINITY
                    {
                        NEG_INFINITY
                    } else {
                        0.0
                    }
                }
                // Rational Rules ll: 0 or -p_outlier
                Likelihood::Rational(p_outlier) => {
                    if trace.rewrites_to(params.max_steps, rhs, Box::new(|_, _| 0.0))
                        == NEG_INFINITY
                    {
                        -p_outlier
                    } else {
                        0.0
                    }
                }
                // trace-sensitive ll: 1-p_trace(h,d)
                // TODO: to be generative, revise to: ll((x,y)|h) = a*(1-p_trace(h,(x,y))) + (1-a)*prior(y)
                Likelihood::Trace => trace.rewrites_to(params.max_steps, rhs, Box::new(|_, _| 0.0)),
                // trace-sensitive ll with string edit distance noise model: (1-p_edit(h,d))
                Likelihood::String { dist, t_max, d_max } => trace.rewrites_to(
                    params.max_steps,
                    rhs,
                    Box::new(move |t1, t2| {
                        UntypedTRS::p_string(t1, t2, dist, t_max, d_max).unwrap_or(NEG_INFINITY)
                    }),
                ),
                // trace-sensitive ll with string edit distance noise model: (1-p_edit(h,d))
                Likelihood::List { dist, t_max, d_max } => trace.rewrites_to(
                    params.max_steps,
                    rhs,
                    Box::new(move |t1, t2| UntypedTRS::p_list(t1, t2, dist, t_max, d_max)),
                ),
            }
        } else {
            NEG_INFINITY
        }
    }

    /// Combine [`pseudo_log_prior`] and [`log_likelihood`], failing early if the
    /// prior is `0.0`.
    ///
    /// [`pseudo_log_prior`]: struct.TRS.html#method.pseudo_log_prior
    /// [`log_likelihood`]: struct.TRS.html#method.log_likelihood
    pub fn posterior(&self, data: &[Rule], params: ModelParams) -> f64 {
        let prior = self.log_prior(params);
        if prior == NEG_INFINITY {
            NEG_INFINITY
        } else {
            let ll = self.log_likelihood(data, params);
            params.p_temp * prior + params.l_temp * ll
        }
    }

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
    ///     parse_rule(&mut sig, "PLUS(x_ ZERO) = x_").expect("parsed rule"),
    ///     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))").expect("parsed rule"),
    /// ];
    ///
    /// let vars = vec![
    ///     ptp![int],
    ///     ptp![int],
    ///     ptp![int],
    /// ];
    ///
    /// for op in sig.operators() {
    ///     println!("{:?}/{}", op.name(), op.arity())
    /// }
    /// for r in &rules {
    ///     println!("{:?}", r.pretty());
    /// }
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], vec![], false, TypeContext::default());
    ///
    /// let mut trs = TRS::new(&lexicon, rules).unwrap();
    ///
    /// assert_eq!(trs.len(), 2);
    ///
    /// let contexts = vec![
    ///     RuleContext {
    ///         lhs: Context::Hole,
    ///         rhs: vec![Context::Hole],
    ///     }
    /// ];
    /// let mut rng = thread_rng();
    /// let atom_weights = (1.0, 1.0, 1.0, 1.0);
    /// let max_size = 50;
    ///
    /// if let Ok(new_trs) = trs.add_rule(&contexts, atom_weights, max_size, &mut rng) {
    ///     assert_eq!(new_trs.len(), 3);
    /// } else {
    ///     assert_eq!(trs.len(), 2);
    /// }
    /// # }
    /// ```
    pub fn add_rule<R: Rng>(
        &self,
        contexts: &[RuleContext],
        atom_weights: (f64, f64, f64, f64),
        max_size: usize,
        rng: &mut R,
    ) -> Result<TRS, SampleError> {
        let mut trs = self.clone();
        let context = contexts.choose(rng).ok_or(SampleError::OptionsExhausted)?;
        let rule = trs
            .lex
            .sample_rule_from_context(context.clone(), atom_weights, true, max_size)
            .drop()?;
        if rule.lhs == rule.rhs().unwrap() {
            return Err(SampleError::Trivial);
        }
        trs.lex.infer_rule(&rule, &mut HashMap::new()).drop()?;
        trs.utrs.push(rule)?;
        Ok(trs)
    }
    /// Regenerate some portion of a rule
    pub fn regenerate_rule<R: Rng>(
        &self,
        atom_weights: (f64, f64, f64, f64),
        max_size: usize,
        rng: &mut R,
    ) -> Result<Vec<TRS>, SampleError> {
        let clause = self.choose_clause(rng)?;
        let new_rules = self.regenerate_helper(&clause, atom_weights, max_size)?;
        let new_trss = new_rules
            .into_iter()
            .filter_map(|new_clause| {
                let mut trs = self.clone();
                let okay = trs.replace(&clause, new_clause).is_ok();
                if okay {
                    Some(trs)
                } else {
                    None
                }
            })
            .collect_vec();
        if new_trss.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(new_trss)
        }
    }
    fn regenerate_helper(
        &self,
        clause: &Rule,
        atom_weights: (f64, f64, f64, f64),
        max_size: usize,
    ) -> Result<Vec<Rule>, SampleError> {
        let rulecontext = RuleContext::from(clause.clone());
        let subcontexts = rulecontext.subcontexts();
        // sample one at random
        let mut rules = Vec::with_capacity(subcontexts.len());
        for subcontext in &subcontexts {
            // replace it with a hole
            let template = rulecontext.replace(&subcontext.1, Context::Hole).unwrap();
            // sample a term from the context
            let trs = self.clone();
            let new_result = trs
                .lex
                .sample_rule_from_context(template, atom_weights, true, max_size)
                .drop();
            if let Ok(new_clause) = new_result {
                if new_clause.lhs != new_clause.rhs().unwrap() {
                    rules.push(new_clause);
                }
            }
        }
        if rules.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(rules)
        }
    }
    /// Delete a rule from the rewrite system if possible. Background knowledge
    /// cannot be deleted.
    pub fn delete_rule(&self) -> Result<Vec<TRS>, SampleError> {
        let background = &self.lex.0.read().expect("poisoned lexicon").background;
        let clauses = self.utrs.clauses();
        let deletable = clauses
            .iter()
            .filter(|c| !background.contains(c))
            .collect_vec();
        if deletable.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            deletable
                .iter()
                .map(|d| {
                    let mut trs = self.clone();
                    trs.utrs.remove_clauses(d)?;
                    Ok(trs)
                })
                .collect()
        }
    }
    /// Replace a subterm of the rule with a variable.
    pub fn replace_term_with_var<R: Rng>(&self, rng: &mut R) -> Result<Vec<TRS>, SampleError> {
        let old_trs = self.clone(); // TODO: cloning is a hack!
        let clause = self.choose_clause(rng)?;
        let mut types = HashMap::new();
        old_trs.lex.infer_rule(&clause, &mut types).drop()?;
        let new_trss = clause
            .lhs
            .subterms()
            .iter()
            .unique_by(|(t, _)| t)
            .filter_map(|(t, p)| {
                // need to indicate that p is from the LHS
                let mut real_p = p.clone();
                real_p.insert(0, 0);
                let mut trs = self.clone();
                let term_type = &types[&real_p];
                let new_var = trs.lex.invent_variable(term_type);
                let new_term = Term::Variable(new_var);
                let new_lhs = clause.lhs.replace_all(t, &new_term);
                let new_rhs = clause.rhs().unwrap().replace_all(t, &new_term);
                Rule::new(new_lhs, vec![new_rhs]).and_then(|new_clause| {
                    let okay = trs.replace(&clause, new_clause).is_ok();
                    if okay {
                        Some(trs)
                    } else {
                        None
                    }
                })
            })
            .collect_vec();
        if new_trss.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(new_trss)
        }
    }
    /// pick a single clause
    fn choose_clause<R: Rng>(&self, rng: &mut R) -> Result<Rule, SampleError> {
        let rules = {
            let num_rules = self.len();
            let num_background = self.num_background_rules();
            &self.utrs.rules[0..(num_rules - num_background)]
        };
        let mut clauses = rules.iter().flat_map(Rule::clauses).collect_vec();
        let idx = (0..clauses.len())
            .choose(rng)
            .ok_or(SampleError::OptionsExhausted)?;
        Ok(clauses.swap_remove(idx))
    }
    pub fn num_background_rules(&self) -> usize {
        self.lex
            .0
            .read()
            .expect("poisoned lexicon")
            .background
            .len()
    }
    pub fn num_learned_rules(&self) -> usize {
        self.len() - self.num_background_rules()
    }
    pub fn replace(
        &mut self,
        old_clause: &Rule,
        new_clause: Rule,
    ) -> Result<&mut TRS, SampleError> {
        self.lex
            .infer_rule(&new_clause, &mut HashMap::new())
            .drop()?;
        self.utrs.replace(0, old_clause, new_clause)?;
        Ok(self)
    }
    /// Move a Rule from one place in the TRS to another at random, excluding the background.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use rand::{thread_rng};
    /// # use polytype::Context as TypeContext;
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
    ///     parse_rule(&mut sig, "PLUS(x_ ZERO) = x_").expect("parsed rule"),
    ///     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))").expect("parsed rule"),
    /// ];
    ///
    /// let vars = vec![
    ///     ptp![int],
    ///     ptp![int],
    ///     ptp![int],
    /// ];
    ///
    /// println!("{:?}", sig.operators());
    /// for op in sig.operators() {
    ///     println!("{:?}/{}", op.name(), op.arity())
    /// }
    /// for r in &rules {
    ///     println!("{:?}", r);
    /// }
    ///
    /// let ctx = TypeContext::default();
    /// let lexicon = Lexicon::from_signature(sig.clone(), ops, vars, vec![], vec![], false, ctx);
    ///
    /// let mut trs = TRS::new(&lexicon, rules).unwrap();
    ///
    /// let pretty_before = trs.to_string();
    ///
    /// let mut rng = thread_rng();
    ///
    /// let new_trs = trs.randomly_move_rule(&mut rng).expect("failed when moving rule");
    ///
    /// assert_ne!(pretty_before, new_trs.to_string());
    /// assert_eq!(new_trs.to_string(), "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_));\nPLUS(x_ ZERO) = x_;");
    /// # }
    /// ```
    pub fn randomly_move_rule<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let mut trs = self.clone();
        let num_rules = self.len();
        let background = &self.lex.0.read().expect("poisoned lexicon").background;
        let num_background = background.len();
        if num_background < num_rules - 1 {
            let i = rng.gen_range(num_background, num_rules);
            let mut j = rng.gen_range(num_background, num_rules);
            while j == i {
                j = rng.gen_range(num_background, num_rules);
            }
            trs.utrs.move_rule(i, j)?;
            Ok(trs)
        } else {
            Err(SampleError::OptionsExhausted)
        }
    }
    /// Selects a rule from the TRS at random, finds all differences in the LHS and RHS,
    /// and makes rules from those differences and inserts them back into the TRS imediately after the background.
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
    ///     parse_rule(&mut sig, "SUCC(PLUS(x_ SUCC(y_))) = SUCC(SUCC(PLUS(x_ y_)))").expect("parsed rule"),
    /// ];
    ///
    /// let vars = vec![
    ///     ptp![int],
    ///     ptp![int],
    ///     ptp![int],
    /// ];
    ///
    /// for op in sig.operators() {
    ///     println!("{:?}/{}", op.name(), op.arity())
    /// }
    /// for r in &rules {
    ///     println!("{:?}", r.pretty());
    /// }
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], vec![], false, TypeContext::default());
    ///
    /// let mut trs = TRS::new(&lexicon, rules).unwrap();
    ///
    /// assert_eq!(trs.len(), 1);
    ///
    /// let mut rng = thread_rng();
    ///
    /// let num_new_trss = trs.local_difference(&mut rng).map(|x| x.len()).ok();
    ///
    /// assert_eq!(Some(2), num_new_trss)
    /// # }
    /// ```
    pub fn local_difference<R: Rng>(&self, rng: &mut R) -> Result<Vec<TRS>, SampleError> {
        let clause = self.choose_clause(rng)?;
        let new_rules = TRS::local_difference_helper(&clause);
        let new_trss = new_rules
            .into_iter()
            .filter_map(|r| {
                let mut trs = self.clone();
                if trs.utrs.replace(0, &clause, r).is_ok() {
                    Some(trs)
                } else {
                    None
                }
            })
            .collect_vec();
        if new_trss.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(new_trss)
        }
    }
    /// Given a rule that has similar terms in the lhs and rhs,
    /// returns a list of rules where each similarity is removed one at a time
    fn local_difference_helper(rule: &Rule) -> Vec<Rule> {
        if let Some(rhs) = rule.rhs() {
            TRS::find_differences(&rule.lhs, &rhs)
                .into_iter()
                .filter_map(|(lhs, rhs)| Rule::new(lhs, vec![rhs]))
                .collect_vec()
        } else {
            vec![]
        }
    }
    // helper for local difference, finds differences in the given lhs and rhs recursively
    fn find_differences(lhs: &Term, rhs: &Term) -> Vec<(Term, Term)> {
        if lhs == rhs {
            return vec![];
        }
        match (lhs, rhs) {
            (Term::Variable(_), _) => vec![], // Variable can't be head of rule
            (
                Term::Application {
                    op: lop,
                    args: largs,
                },
                Term::Application {
                    op: rop,
                    args: rargs,
                },
            ) if lop == rop && !largs.is_empty() => largs
                .iter()
                .zip(rargs)
                .flat_map(|(l, r)| TRS::find_differences(l, r))
                .chain(once((lhs.clone(), rhs.clone())))
                .collect_vec(),
            _ => vec![(lhs.clone(), rhs.clone())],
        }
    }
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
    ///     println!("{:?}/{}", op.name(), op.arity())
    /// }
    /// for r in &rules {
    ///     println!("{:?}", r.pretty());
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
    /// Given a list of `Rule`s considered to be data, add one datum as a rule.
    pub fn add_exception(&self, data: &[Rule]) -> Result<Vec<TRS>, SampleError> {
        let results = data
            .iter()
            .filter_map(|d| {
                let mut trs = self.clone();
                if trs.utrs.push(d.clone()).is_ok() {
                    Some(trs)
                } else {
                    None
                }
            })
            .collect_vec();
        if results.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(results)
        }
    }
    pub fn generalize(&self, data: &[Rule]) -> Result<Vec<TRS>, SampleError> {
        let mut trs = self.clone();
        let mut all_rules = trs.utrs.clauses();
        all_rules.extend_from_slice(&trs.novel_rules(data));
        let (lhs_context, clauses) = TRS::find_lhs_context(&all_rules)?;
        let (rhs_context, _) = TRS::find_rhs_context(&clauses)?;
        let new_rules = TRS::generalize_clauses(&trs.lex, &lhs_context, &rhs_context, &clauses)?;
        trs.remove_clauses(&clauses)?;
        trs.prepend_clauses(new_rules)?;
        Ok(vec![trs])
    }
    fn novel_rules(&self, rules: &[Rule]) -> Vec<Rule> {
        rules
            .iter()
            .cloned()
            .filter(|r| !self.contains(r))
            .collect()
    }
    fn find_lhs_context(clauses: &[Rule]) -> Result<(Context, Vec<Rule>), SampleError> {
        TRS::find_shared_context(clauses, |c| c.lhs.clone(), 1)
    }
    fn find_rhs_context(clauses: &[Rule]) -> Result<(Context, Vec<Rule>), SampleError> {
        TRS::find_shared_context(clauses, |c| c.rhs().unwrap(), 3) // constant is a HACK
    }
    fn find_shared_context<T>(
        clauses: &[Rule],
        f: T,
        max_holes: usize,
    ) -> Result<(Context, Vec<Rule>), SampleError>
    where
        T: Fn(&Rule) -> Term,
    {
        // Collect all contexts and their witnesses.
        let mut contexts: Vec<(Context, &Rule)> = Vec::new();
        for clause in clauses {
            for context in f(clause).contexts(max_holes) {
                contexts.push((context, &clause));
            }
        }
        // Dump them into a HashMap.
        let mut map: HashMap<&Context, Vec<&Rule>> = HashMap::new();
        for (context, clause) in &contexts {
            let key: &Context = map
                .keys()
                .find(|k| Context::alpha(context, k).is_some())
                .unwrap_or(&context);
            (*map.entry(key).or_insert_with(|| vec![])).push(clause);
        }
        // Pick the largest context witnessed by >= 2 clauses.
        map.iter()
            .max_by_key(|(k, v)| ((v.len() > 1) as usize) * (k.size() - k.holes().len()))
            .map(|(&k, v)| (k.clone(), v.iter().map(|&x| x.clone()).collect_vec()))
            .ok_or_else(|| SampleError::OptionsExhausted)
    }
    // The workhorse behind generalization.
    fn generalize_clauses(
        lex: &Lexicon,
        lhs_context: &Context,
        rhs_context: &Context,
        clauses: &[Rule],
    ) -> Result<Vec<Rule>, SampleError> {
        // Create the LHS.
        let (lhs, lhs_place, var) =
            TRS::fill_hole_with_variable(lex, &lhs_context).ok_or(SampleError::Subterm)?;
        // Fill the RHS context and create subproblem rules.
        let mut rhs = rhs_context.clone();
        let mut new_rules: Vec<Rule> = vec![];
        for rhs_place in &rhs_context.holes() {
            // Collect term, type, and variable information from each clause.
            let (types, terms, vars) =
                TRS::collect_information(lex, &lhs, &lhs_place, rhs_place, clauses, &var)?;
            // Infer the type for this place.
            let return_tp = TRS::compute_place_type(lex, &types)?;
            // Create the new operator for this place. TODO HACK: make applicative parameterizable.
            let new_op = TRS::new_operator(lex, true, &vars, &return_tp)?;
            // Create the rules expressing subproblems for this place.
            for (lhs_term, rhs_term) in &terms {
                let new_rule = TRS::new_rule(lex, &new_op, lhs_term, rhs_term, &var, &vars)?;
                new_rules.push(new_rule);
            }
            // Fill the hole at this place in the RHS.
            rhs = TRS::fill_next_hole(lex, &rhs, rhs_place, new_op, vars)?;
        }
        let rhs_term = rhs.to_term().map_err(|_| SampleError::Subterm)?;
        // Create the generalized rule.
        new_rules.push(Rule::new(lhs, vec![rhs_term]).ok_or(SampleError::Subterm)?);
        Ok(new_rules)
    }
    fn fill_next_hole(
        lex: &Lexicon,
        rhs: &Context,
        place: &[usize],
        new_op: Operator,
        vars: Vec<Variable>,
    ) -> Result<Context, SampleError> {
        let app = lex.has_op(Some("."), 2).map_err(|_| SampleError::Subterm)?;
        let mut subctx = Context::from(Atom::from(new_op));
        for var in vars {
            subctx = Context::Application {
                op: app.clone(),
                args: vec![subctx, Context::from(Atom::from(var))],
            };
        }
        rhs.replace(place, subctx).ok_or(SampleError::Subterm)
    }
    fn collect_information<'a>(
        lex: &Lexicon,
        lhs: &Term,
        lhs_place: &[usize],
        rhs_place: &[usize],
        clauses: &'a [Rule],
        var: &Variable,
    ) -> Result<(Vec<Type>, Vec<(&'a Term, Term)>, Vec<Variable>), SampleError> {
        let mut terms = vec![];
        let mut types = vec![];
        let mut vars = vec![var.clone()];
        for clause in clauses {
            let rhs = clause.rhs().ok_or(SampleError::Subterm)?;
            println!("rhs: {}", rhs.pretty());
            let lhs_subterm = clause.lhs.at(lhs_place).ok_or(SampleError::Subterm)?;
            println!("lhs_subterm: {}", lhs_subterm.pretty());
            let rhs_subterm = rhs.at(rhs_place).ok_or(SampleError::Subterm)?;
            println!("rhs_subterm: {}", rhs_subterm.pretty());
            let mut map = HashMap::new();
            lex.infer_term(&rhs, &mut map).drop()?;
            println!("type: {}", map[rhs_place]);
            types.push(map[rhs_place].clone());
            println!("master lhs: {}", lhs.pretty());
            println!("clause lhs: {}", clause.lhs.pretty());
            let alpha = Term::pmatch(vec![(&lhs, &clause.lhs)]).ok_or(SampleError::Subterm)?;
            println!("alpha succeeded");
            for var in &rhs_subterm.variables() {
                let var_term = Term::Variable(var.clone());
                if let Some((&k, _)) = alpha.iter().find(|(_, &v)| *v == var_term) {
                    vars.push(k.clone())
                } else {
                    return Err(SampleError::Subterm);
                }
            }
            terms.push((lhs_subterm, rhs_subterm.clone()));
        }
        Ok((types, terms, vars))
    }
    fn compute_place_type(lex: &Lexicon, types: &[Type]) -> Result<Type, SampleError> {
        let return_tp = lex.fresh_type_variable();
        for tp in types {
            lex.unify(&return_tp, tp)
                .map_err(|_| SampleError::Subterm)?;
        }
        Ok(return_tp.apply(&lex.0.read().expect("poisoned lexicon").ctx))
    }
    // Patch a one-hole `Context` with a `Variable`.
    fn fill_hole_with_variable(
        lex: &Lexicon,
        context: &Context,
    ) -> Option<(Term, Vec<usize>, Variable)> {
        // Confirm that there's a single hole and find its place.
        let mut holes = context.holes();
        if holes.len() != 1 {
            return None;
        }
        let hole = holes.pop()?;
        // Create a variable whose type matches the hole.
        let mut tps = HashMap::new();
        lex.infer_context(context, &mut tps).drop().ok()?;
        let new_var = lex.invent_variable(&tps[&hole]);
        // Replace the hole with the new variable.
        let filled_context = context.replace(&hole, Context::from(Atom::from(new_var.clone())))?;
        let term = filled_context.to_term().ok()?;
        // Return the term, the hole, and its replacement.
        Some((term, hole, new_var))
    }
    // Create a new `Operator` whose type is consistent with `Vars`.
    fn new_operator(
        lex: &Lexicon,
        applicative: bool,
        vars: &[Variable],
        return_tp: &Type,
    ) -> Result<Operator, SampleError> {
        // Construct the name.
        let name = None;
        // Construct the arity.
        let arity = (!applicative as u32) * (vars.len() as u32);
        // Construct the type.
        let mut tp = return_tp.clone();
        for var in vars.iter().rev() {
            let schema = lex.infer_var(var)?;
            tp = Type::arrow(lex.instantiate(&schema), tp);
        }
        // Create the new variable.
        Ok(lex.invent_operator(name, arity, &tp))
    }
    // Create a new rule setting up a generalization subproblem.
    fn new_rule(
        lex: &Lexicon,
        op: &Operator,
        lhs_arg: &Term,
        rhs: &Term,
        var: &Variable,
        vars: &[Variable],
    ) -> Result<Rule, SampleError> {
        let mut lhs = Term::apply(op.clone(), vec![]).ok_or(SampleError::Subterm)?;
        let app = lex.has_op(Some("."), 2).map_err(|_| SampleError::Subterm)?;
        for v in vars {
            let arg = if v == var {
                lhs_arg.clone()
            } else {
                Term::Variable(v.clone())
            };
            lhs = Term::apply(app.clone(), vec![lhs, arg]).ok_or(SampleError::Subterm)?;
        }
        Rule::new(lhs, vec![rhs.clone()]).ok_or(SampleError::Subterm)
    }
    fn contains(&self, rule: &Rule) -> bool {
        self.utrs.get_clause(rule).is_some()
    }
    fn remove_clauses(&mut self, rules: &[Rule]) -> Result<(), SampleError> {
        for rule in rules {
            if self.contains(rule) {
                self.utrs.remove_clauses(&rule)?;
            }
        }
        Ok(())
    }
    fn prepend_clauses(&mut self, rules: Vec<Rule>) -> Result<(), SampleError> {
        self.utrs
            .pushes(rules)
            .map(|_| ())
            .map_err(SampleError::from)
    }
}
impl fmt::Display for TRS {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let true_len = self.utrs.len()
            - self
                .lex
                .0
                .read()
                .expect("poisoned lexicon")
                .background
                .len();
        let trs_str = self
            .utrs
            .rules
            .iter()
            .take(true_len)
            .map(|r| format!("{};", r.display()))
            .join("\n");

        write!(f, "{}", trs_str)
    }
}

#[cfg(test)]
mod tests {
    use super::TRS;
    use itertools::Itertools;
    use polytype::Context as TypeContext;
    use std::collections::HashMap;
    use term_rewriting::{Atom, Context, Rule};
    use trs::parser::{parse_context, parse_lexicon, parse_rule, parse_term, parse_trs};

    #[test]
    fn find_lhs_context_test() {
        let lex = parse_lexicon(
            &[
                "+/2: INT -> INT -> INT;",
                " */2: INT -> INT -> INT;",
                " ^/2: INT -> INT -> INT;",
                " 0/0: INT; 1/0: INT; 2/0: INT;",
                " 3/0: INT; 4/0: INT; 6/0: INT;",
                " 9/0: INT;",
            ]
            .join(" "),
            "",
            "",
            true,
            TypeContext::default(),
        )
        .unwrap();
        let trs = parse_trs(
            "^(+(x_ 1) 2) = +(^(x_ 2) +(*(2 x_) 1)); ^(+(x_ 2) 2) = +(^(x_ 2) +(*(4 x_) 4)); ^(+(x_ 3) 2) = +(^(x_ 2) +(*(6 x_) 9)); +(x_ 0) = x_; +(0 x_) = x_;",
            &lex,
        )
            .expect("parsed trs");
        let clauses = trs.utrs.clauses();
        let (context, clauses) = TRS::find_lhs_context(&clauses).unwrap();

        assert_eq!("^(+(x_, [!]), 2)", context.pretty());
        assert_eq!(3, clauses.len());
    }

    #[test]
    fn find_rhs_context_test() {
        let lex = parse_lexicon(
            &[
                "+/2: INT -> INT -> INT;",
                " */2: INT -> INT -> INT;",
                " ^/2: INT -> INT -> INT;",
                " 0/0: INT; 1/0: INT; 2/0: INT;",
                " 3/0: INT; 4/0: INT; 6/0: INT;",
                " 9/0: INT;",
            ]
            .join(" "),
            "",
            "",
            true,
            TypeContext::default(),
        )
        .unwrap();
        let clauses = vec![
            parse_rule("^(+(x_ 1) 2) = +(^(x_ 2) +(*(2 x_) 1))", &lex).expect("parsed rule 1"),
            parse_rule("^(+(x_ 2) 2) = +(^(x_ 2) +(*(4 x_) 4))", &lex).expect("parsed rule 2"),
            parse_rule("^(+(x_ 3) 2) = +(^(x_ 2) +(*(6 x_) 9))", &lex).expect("parsed rule 3"),
        ];
        let (context, clauses) = TRS::find_rhs_context(&clauses).unwrap();

        assert_eq!("+(^(x_, 2), +(*([!], x_), [!]))", context.pretty());
        assert_eq!(3, clauses.len());
    }

    #[test]
    fn fill_hole_with_variable_test() {
        let lex = parse_lexicon(
            &[
                "+/2: INT -> INT -> INT;",
                " */2: INT -> INT -> INT;",
                " ^/2: INT -> INT -> INT;",
                " 0/0: INT; 1/0: INT; 2/0: INT;",
                " 3/0: INT; 4/0: INT; 6/0: INT;",
                " 9/0: INT;",
            ]
            .join(" "),
            "",
            "",
            true,
            TypeContext::default(),
        )
        .unwrap();
        let context = parse_context("^(+(x_ [!]) 2)", &lex).expect("parsed context");
        let (term, place, var) = TRS::fill_hole_with_variable(&lex, &context).unwrap();
        let tp = lex
            .infer_context(&Context::from(Atom::from(var)), &mut HashMap::new())
            .drop()
            .unwrap();

        assert_eq!("^(+(x_, var1_), 2)", term.pretty());
        assert_eq!(vec![0, 1], place);
        assert_eq!("INT", tp.to_string());
    }

    #[test]
    fn new_operator_test() {
        let lex = parse_lexicon(
            "+/0: INT -> INT -> INT; */0: INT -> INT -> INT; ^/0: INT -> INT -> INT; ./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
            "",
            "",
            true,
            TypeContext::default(),
        )
        .unwrap();
        let applicative = true;
        let vars = &[
            lex.invent_variable(&tp!(INT)),
            lex.invent_variable(&tp!(LIST)),
        ];
        let return_tp = tp!(LIST);
        let op = TRS::new_operator(&lex, applicative, vars, &return_tp).unwrap();
        let tp = lex
            .infer_context(&Context::from(Atom::from(op.clone())), &mut HashMap::new())
            .drop()
            .unwrap();

        assert_eq!("op4", op.display());
        assert_eq!(0, op.arity());
        assert_eq!("INT → LIST → LIST", tp.to_string());
    }

    #[test]
    fn new_rule_test() {
        let lex = parse_lexicon(
            &[
                "+/2: INT -> INT -> INT;",
                " */2: INT -> INT -> INT;",
                " ^/2: INT -> INT -> INT;",
                "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
                " 0/0: INT; 1/0: INT; 2/0: INT;",
                " 3/0: INT; 4/0: INT; 6/0: INT;",
                " 9/0: INT;",
            ]
            .join(" "),
            "",
            "",
            true,
            TypeContext::default(),
        )
        .unwrap();
        let op = lex.invent_operator(Some("F".to_string()), 0, &tp![@arrow[tp!(INT), tp!(INT)]]);
        let lhs_arg = parse_term("1", &lex).unwrap();
        let rhs = parse_term("2", &lex).unwrap();
        let var = lex.invent_variable(&tp![INT]);
        let vars = vec![var.clone()];
        let rule = TRS::new_rule(&lex, &op, &lhs_arg, &rhs, &var, &vars).unwrap();

        assert_eq!("F 1 = 2", rule.pretty());
    }

    #[test]
    fn generalize_clauses_test() {
        let lex = parse_lexicon(
            &[
                "+/2: INT -> INT -> INT;",
                " */2: INT -> INT -> INT;",
                " ^/2: INT -> INT -> INT;",
                "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
                " 0/0: INT; 1/0: INT; 2/0: INT;",
                " 3/0: INT; 4/0: INT; 6/0: INT;",
                " 9/0: INT;",
            ]
            .join(" "),
            "",
            "",
            true,
            TypeContext::default(),
        )
        .unwrap();
        let trs = parse_trs(
            "^(+(x_ 1) 2) = +(^(x_ 2) +(*(2 x_) 1)); ^(+(x_ 2) 2) = +(^(x_ 2) +(*(4 x_) 4)); ^(+(x_ 3) 2) = +(^(x_ 2) +(*(6 x_) 9)); +(x_ 0) = x_; +(0 x_) = x_;",
            &lex,
        )
            .expect("parsed trs");
        let all_clauses = trs.utrs.clauses();
        let (lhs_context, clauses) = TRS::find_lhs_context(&all_clauses).unwrap();
        let (rhs_context, _) = TRS::find_rhs_context(&clauses).unwrap();
        let rules =
            TRS::generalize_clauses(&trs.lex, &lhs_context, &rhs_context, &clauses).unwrap();
        let rule_string = rules.iter().map(Rule::pretty).join("\n");

        assert_eq!(
            "op11 1 = 2\nop11 2 = 4\nop11 3 = 6\nop12 1 = 1\nop12 2 = 4\nop12 3 = 9\n^(+(x_, var5_), 2) = +(^(x_, 2), +(*(op11 var5_, x_), op12 var5_))",
            rule_string
        );
    }

    #[test]
    fn generalize_test() {
        let lex = parse_lexicon(
            &[
                "+/2: INT -> INT -> INT;",
                " */2: INT -> INT -> INT;",
                " ^/2: INT -> INT -> INT;",
                "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
                " 0/0: INT; 1/0: INT; 2/0: INT;",
                " 3/0: INT; 4/0: INT; 6/0: INT;",
                " 9/0: INT;",
            ]
            .join(" "),
            "",
            "",
            true,
            TypeContext::default(),
        )
        .unwrap();
        let trs = parse_trs(
            "^(+(x_ 1) 2) = +(^(x_ 2) +(*(2 x_) 1)); ^(+(x_ 2) 2) = +(^(x_ 2) +(*(4 x_) 4)); ^(+(x_ 3) 2) = +(^(x_ 2) +(*(6 x_) 9)); +(x_ 0) = x_; +(0 x_) = x_;",
            &lex,
        )
            .expect("parsed trs");
        let trs = trs.generalize(&[]).unwrap().pop().unwrap();
        let rule_string = trs.utrs.rules.iter().map(Rule::pretty).join("\n");

        assert_eq!(
            "op11 1 = 2\nop11 2 = 4\nop11 3 = 6\nop12 1 = 1\nop12 2 = 4\nop12 3 = 9\n^(+(x_, var5_), 2) = +(^(x_, 2), +(*(op11 var5_, x_), op12 var5_))\n+(x_, 0) = x_\n+(0, x_) = x_",
            rule_string
        );
    }
}

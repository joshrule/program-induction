//! Polymorphically typed (Hindley-Milner) First-Order Term Rewriting Systems (no abstraction)
//!
//! Much thanks to:
//! - https://github.com/rob-smallshire/hindley-milner-python
//! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! - (TAPL; Pierce, 2002, ch. 22)

use itertools::Itertools;
use polytype::Context as TypeContext;
use rand::{seq::IteratorRandom, seq::SliceRandom, Rng};
use std::collections::HashMap;
use std::f64::NEG_INFINITY;
use std::fmt;
use std::iter::once;
use term_rewriting::trace::Trace;
use term_rewriting::{Context, Rule, RuleContext, Term, TRS as UntypedTRS};

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
        trs.lex.infer_rule(&rule, &mut HashMap::new())?;
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
        old_trs.lex.infer_rule(&clause, &mut types)?;
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
            let background = &self.lex.0.read().expect("poisoned lexicon").background;
            let num_background = background.len();
            &self.utrs.rules[0..(num_rules - num_background)]
        };
        let mut clauses = rules.iter().flat_map(Rule::clauses).collect_vec();
        let idx = (0..clauses.len())
            .choose(rng)
            .ok_or(SampleError::OptionsExhausted)?;
        Ok(clauses.swap_remove(idx))
    }
    pub fn replace(
        &mut self,
        old_clause: &Rule,
        new_clause: Rule,
    ) -> Result<&mut TRS, SampleError> {
        self.lex.infer_rule(&new_clause, &mut HashMap::new())?;
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

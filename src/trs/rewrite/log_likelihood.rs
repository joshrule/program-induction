use std::collections::HashMap;
use std::{convert::TryInto, f64::NEG_INFINITY};
use term_rewriting::{trace::Trace, Rule, Term, TRS as UntypedTRS};
use trs::{Environment, Likelihood, SingleLikelihood, TRS};
use utils::logsumexp;

impl<'a, 'b> TRS<'a, 'b> {
    /// A log likelihood for a `TRS`: the probability of `data`'s RHSs appearing
    /// in [`term_rewriting::Trace`]s rooted at its LHSs.
    ///
    /// [`term_rewriting::Trace`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/trace/struct.Trace.html
    pub fn log_likelihood(
        &self,
        data: &[Rule],
        input: Option<&Term>,
        evals: &mut HashMap<Rule, f64>,
        likelihood: Likelihood,
    ) -> f64 {
        let n_data = data.len() + input.is_some() as usize;
        let ll = data
            .iter()
            .enumerate()
            .map(|(i, datum)| {
                let weight = likelihood
                    .decay
                    .powi(n_data.saturating_sub(i + 1).try_into().unwrap());
                let ll = evals
                    .entry(datum.clone())
                    .or_insert_with(|| self.single_log_likelihood(datum, likelihood));
                weight.ln() + *ll
            })
            .sum();
        input
            .map(|term| ll + self.partial_single_log_likelihood(term, likelihood))
            .unwrap_or(ll)
    }

    /// Compute the log likelihood when all you have is a single input.
    pub(crate) fn partial_single_log_likelihood(
        &self,
        input: &Term,
        likelihood: Likelihood,
    ) -> f64 {
        match likelihood.single {
            SingleLikelihood::List { alpha, .. } => {
                let utrs = self.full_utrs();
                let sig = &self.lex.0.signature;
                let mut trace = Trace::new(
                    &utrs,
                    &sig,
                    input,
                    likelihood.p_observe,
                    likelihood.max_size,
                    likelihood.max_depth,
                    likelihood.strategy,
                );
                trace.rewrite(likelihood.max_steps);
                // walk the trace and weight each output
                let lps = trace
                    .root()
                    .iter()
                    .map(|n| {
                        if UntypedTRS::convert_list_to_string(&n.term(), &sig.deep_copy()).is_some()
                        {
                            // (1-a)p
                            // TODO: would be nice to have a generative story here
                            n.log_p() + (1.0 - alpha).ln()
                        } else {
                            // ap
                            n.log_p() + alpha.ln()
                        }
                    })
                    .collect::<Vec<_>>();
                logsumexp(&lps)
            }
            // Partials aren't implemented for any other likelihoods yet :-)
            _ => 0.0,
        }
    }

    /// Compute the log likelihood for a single datum.
    pub(crate) fn single_log_likelihood(&self, datum: &Rule, likelihood: Likelihood) -> f64 {
        let utrs = self.full_utrs();
        let sig = &self.lex.0.signature;
        if let Some(ref rhs) = datum.rhs() {
            let mut trace = Trace::new(
                &utrs,
                &sig,
                &datum.lhs,
                likelihood.p_observe,
                likelihood.max_size,
                likelihood.max_depth,
                likelihood.strategy,
            );
            match likelihood.single {
                // Binary ll: 0 or -\infty
                SingleLikelihood::Binary => {
                    if trace.rewrites_to(likelihood.max_steps, rhs, |_, _| 0.0) == NEG_INFINITY {
                        NEG_INFINITY
                    } else {
                        0.0
                    }
                }
                // Rational Rules ll: 0 or -p_outlier
                SingleLikelihood::Rational(p_outlier) => {
                    if trace.rewrites_to(likelihood.max_steps, rhs, |_, _| 0.0) == NEG_INFINITY {
                        -p_outlier
                    } else {
                        0.0
                    }
                }
                // trace-sensitive ll: 1-p_trace(h,d)
                // TODO: to be generative, revise to: ll((x,y)|h) = a*(1-p_trace(h,(x,y))) + (1-a)*prior(y)
                SingleLikelihood::Trace => trace.rewrites_to(likelihood.max_steps, rhs, |_, _| 0.0),
                // trace-sensitive ll with string edit distance noise model: (1-p_edit(h,d))
                SingleLikelihood::String { dist, t_max, d_max } => {
                    trace.rewrites_to(likelihood.max_steps, rhs, move |t1, t2| {
                        UntypedTRS::p_string(t1, t2, dist, t_max, d_max, &sig)
                            .unwrap_or(NEG_INFINITY)
                    })
                }
                // trace-sensitive ll with string edit distance noise model: (1-p_edit(h,d))
                SingleLikelihood::List {
                    alpha,
                    atom_weights,
                    dist,
                    t_max,
                    d_max,
                } => {
                    let mut ctx = self.lex.0.ctx.clone();
                    let mut env = Environment::from_vars(&datum.lhs.variables(), &mut ctx);
                    let mut types = HashMap::new();
                    self.lex
                        .infer_term(&datum.lhs, &mut types, &mut env, &mut ctx)
                        .ok();
                    env.invent = false;
                    let schema = ptp!(list);
                    let p_sample = self
                        .lex
                        .logprior_term(rhs, &schema, atom_weights, &mut env, &mut ctx)
                        .unwrap_or_else(|_| NEG_INFINITY);
                    let p_rewrite = trace.rewrites_to(likelihood.max_steps, rhs, move |t1, t2| {
                        UntypedTRS::p_list(t1, t2, dist, t_max, d_max, &sig)
                    });
                    logsumexp(&[alpha.ln() + p_sample, (1.0 - alpha).ln() + p_rewrite])
                }
            }
        } else {
            NEG_INFINITY
        }
    }
}

use std::collections::HashMap;
use std::f64::NEG_INFINITY;
use term_rewriting::{trace::Trace, Rule, Term, TRS as UntypedTRS};
use trs::{Datum, Eval, Likelihood, SingleLikelihood, TRS};
use utils::logsumexp;

impl<'a, 'b> TRS<'a, 'b> {
    /// A log likelihood for a `TRS`: the probability of `data`'s RHSs appearing
    /// in [`term_rewriting::Trace`]s rooted at its LHSs.
    ///
    /// [`term_rewriting::Trace`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/trace/struct.Trace.html
    pub fn log_likelihood(
        &self,
        data: &[&'b Datum],
        evals: &mut HashMap<&'b Datum, Eval>,
        likelihood: Likelihood,
    ) -> f64 {
        data.iter()
            .rev()
            .enumerate()
            .map(|(i, datum)| {
                let weight = likelihood.decay.powi(i as i32);
                let ll = evals
                    .entry(datum)
                    .or_insert_with(|| self.single_log_likelihood(datum, likelihood));
                weight.ln() + ll.likelihood()
            })
            .sum()
    }

    pub(crate) fn single_log_likelihood(&self, datum: &Datum, likelihood: Likelihood) -> Eval {
        match datum {
            Datum::Full(ref rule) => self.full_single_log_likelihood(rule, likelihood),
            Datum::Partial(ref term) => self.partial_single_log_likelihood(term, likelihood),
        }
    }

    /// Compute the log likelihood when all you have is a single input.
    pub(crate) fn partial_single_log_likelihood(
        &self,
        input: &Term,
        likelihood: Likelihood,
    ) -> Eval {
        match likelihood.single {
            SingleLikelihood::List { alpha, .. } => {
                let utrs = self.full_utrs();
                let sig = &self.lex.lex.sig;
                let trace = Trace::new(
                    &utrs,
                    &sig,
                    input,
                    likelihood.p_observe,
                    likelihood.max_steps,
                    likelihood.max_size,
                    likelihood.strategy,
                );
                // walk the trace and weight each output
                let mut generalizes = false;
                let lps = trace
                    .iter()
                    .map(|n| {
                        if UntypedTRS::convert_list_to_string(trace[n].term(), &mut sig.clone())
                            .is_some()
                        {
                            // (1-a)p
                            // TODO: would be nice to have a generative story here
                            generalizes = generalizes || trace[n].log_p() > NEG_INFINITY;
                            trace[n].log_p() + (1.0 - alpha).ln()
                        } else {
                            // ap
                            trace[n].log_p() + alpha.ln()
                        }
                    })
                    .collect::<Vec<_>>();
                Eval::Partial(logsumexp(&lps), generalizes)
            }
            // Partials aren't implemented for any other likelihoods yet :-)
            _ => Eval::Partial(0.0, true),
        }
    }

    /// Compute the log likelihood for a single datum.
    pub(crate) fn full_single_log_likelihood(&self, datum: &Rule, likelihood: Likelihood) -> Eval {
        let utrs = self.full_utrs();
        let sig = &self.lex.lex.sig;
        if let Some(ref rhs) = datum.rhs() {
            let mut trace = Trace::new(
                &utrs,
                &sig,
                &datum.lhs,
                likelihood.p_observe,
                likelihood.max_steps,
                likelihood.max_size,
                likelihood.strategy,
            );
            match likelihood.single {
                // Binary ll: 0 or -\infty
                SingleLikelihood::Binary => {
                    if trace.rewrites_to(rhs, |_, _| 0.0) == NEG_INFINITY {
                        Eval::Full(NEG_INFINITY)
                    } else {
                        Eval::Full(0.0)
                    }
                }
                // Rational Rules ll: 0 or -p_outlier
                SingleLikelihood::Rational(p_outlier) => {
                    if trace.rewrites_to(rhs, |_, _| 0.0) == NEG_INFINITY {
                        Eval::Full(-p_outlier)
                    } else {
                        Eval::Full(0.0)
                    }
                }
                // trace-sensitive ll: 1-p_trace(h,d)
                // TODO: to be generative, revise to: ll((x,y)|h) = a*(1-p_trace(h,(x,y))) + (1-a)*prior(y)
                SingleLikelihood::Trace => Eval::Full(trace.rewrites_to(rhs, |_, _| 0.0)),
                // trace-sensitive ll with string edit distance noise model: (1-p_edit(h,d))
                SingleLikelihood::String { dist, t_max, d_max } => {
                    Eval::Full(trace.rewrites_to(rhs, move |t1, t2| {
                        UntypedTRS::p_string(t1, t2, dist, t_max, d_max, &sig)
                            .unwrap_or(NEG_INFINITY)
                    }))
                }
                // trace-sensitive ll with string edit distance noise model: (1-p_edit(h,d))
                SingleLikelihood::List {
                    alpha,
                    atom_weights,
                    dist,
                    t_max,
                    d_max,
                } => {
                    if let Ok(mut env) = self.lex.infer_term(&datum.lhs) {
                        env.invent = false;
                        let list = self.lex.lex.ctx.intern_name("list");
                        let tp = self.lex.lex.ctx.intern_tcon(list, &[]);
                        let p_sample = env
                            .logprior_term(rhs, tp, atom_weights)
                            .unwrap_or(NEG_INFINITY);
                        let p_rewrite = trace.rewrites_to(rhs, move |t1, t2| {
                            UntypedTRS::p_list(t1, t2, dist, t_max, d_max, &sig)
                        });
                        Eval::Full(logsumexp(&[
                            alpha.ln() + p_sample,
                            (1.0 - alpha).ln() + p_rewrite,
                        ]))
                    } else {
                        Eval::Full(NEG_INFINITY)
                    }
                }
            }
        } else {
            Eval::Full(NEG_INFINITY)
        }
    }
}

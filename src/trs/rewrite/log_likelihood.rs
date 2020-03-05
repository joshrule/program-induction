use std::{convert::TryInto, f64::NEG_INFINITY};
use term_rewriting::{trace::Trace, Rule, TRS as UntypedTRS};

use trs::{Likelihood, SingleLikelihood, TRS};

impl<'a, 'b> TRS<'a, 'b> {
    /// A log likelihood for a `TRS`: the probability of `data`'s RHSs appearing
    /// in [`term_rewriting::Trace`]s rooted at its LHSs.
    ///
    /// [`term_rewriting::Trace`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/trace/struct.Trace.html
    pub fn log_likelihood(&self, data: &[Rule], likelihood: Likelihood) -> f64 {
        let n_data = data.len();
        data.iter()
            .enumerate()
            .map(|(i, datum)| {
                let weight = likelihood
                    .decay
                    .powi(n_data.saturating_sub(i + 1).try_into().unwrap());
                weight.ln() + self.single_log_likelihood(datum, likelihood)
            })
            .sum()
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
                SingleLikelihood::List { dist, t_max, d_max } => {
                    trace.rewrites_to(likelihood.max_steps, rhs, move |t1, t2| {
                        UntypedTRS::p_list(t1, t2, dist, t_max, d_max, &sig)
                    })
                }
            }
        } else {
            NEG_INFINITY
        }
    }
}

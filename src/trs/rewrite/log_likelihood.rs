use std::f64::NEG_INFINITY;
use term_rewriting::{trace::Trace, Rule, TRS as UntypedTRS};

use super::{Likelihood, ModelParams, TRS};

impl TRS {
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
        let sig = &self.lex.0.read().expect("poisoned lexicon").signature;
        if let Some(ref rhs) = datum.rhs() {
            let mut trace = Trace::new(
                &self.utrs,
                &sig,
                &datum.lhs,
                params.p_observe,
                params.max_size,
                params.max_depth,
                params.strategy,
            );
            match params.likelihood {
                // Binary ll: 0 or -\infty
                Likelihood::Binary => {
                    if trace.rewrites_to(params.max_steps, rhs, |_, _| 0.0) == NEG_INFINITY {
                        NEG_INFINITY
                    } else {
                        0.0
                    }
                }
                // Rational Rules ll: 0 or -p_outlier
                Likelihood::Rational(p_outlier) => {
                    if trace.rewrites_to(params.max_steps, rhs, |_, _| 0.0) == NEG_INFINITY {
                        -p_outlier
                    } else {
                        0.0
                    }
                }
                // trace-sensitive ll: 1-p_trace(h,d)
                // TODO: to be generative, revise to: ll((x,y)|h) = a*(1-p_trace(h,(x,y))) + (1-a)*prior(y)
                Likelihood::Trace => trace.rewrites_to(params.max_steps, rhs, |_, _| 0.0),
                // trace-sensitive ll with string edit distance noise model: (1-p_edit(h,d))
                Likelihood::String { dist, t_max, d_max } => {
                    trace.rewrites_to(params.max_steps, rhs, move |t1, t2| {
                        UntypedTRS::p_string(t1, t2, dist, t_max, d_max, sig)
                            .unwrap_or(NEG_INFINITY)
                    })
                }
                // trace-sensitive ll with string edit distance noise model: (1-p_edit(h,d))
                Likelihood::List { dist, t_max, d_max } => {
                    trace.rewrites_to(params.max_steps, rhs, move |t1, t2| {
                        UntypedTRS::p_list(t1, t2, dist, t_max, d_max, sig)
                    })
                }
            }
        } else {
            NEG_INFINITY
        }
    }
}

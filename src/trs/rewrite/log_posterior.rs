use std::f64::NEG_INFINITY;
use term_rewriting::{Rule, Term};

use trs::{rewrite::TRS, ModelParams};

impl<'a, 'b> TRS<'a, 'b> {
    /// Combine [`log_prior`] and [`log_likelihood`], failing early if the
    /// prior is `0.0`.
    ///
    /// [`log_prior`]: struct.TRS.html#method.log_prior
    /// [`log_likelihood`]: struct.TRS.html#method.log_likelihood
    pub fn log_posterior(
        &self,
        data: &[Rule],
        input: Option<&Term>,
        t: f64,
        params: ModelParams,
    ) -> f64 {
        let prior = self.log_prior(params.prior);
        if prior == NEG_INFINITY {
            NEG_INFINITY
        } else {
            let ll = self.log_likelihood(data, input, params.likelihood);
            let temperature = params.schedule.temperature(t);
            (params.p_temp * prior + params.l_temp * ll) / temperature
        }
    }
}

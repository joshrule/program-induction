use std::f64::NEG_INFINITY;
use trs::{Prior, TRS};
use utils::{block_generative_logpdf, fail_geometric_logpdf};

impl<'a, 'b> TRS<'a, 'b> {
    /// A pseudo log prior for a `TRS`: the negative [`size`] of the `TRS`
    /// scaled by some cost per token.
    ///
    /// [`size`]: struct.TRS.html#method.size
    pub fn log_prior(&self, prior: Prior) -> f64 {
        match prior {
            Prior::Size(p_token) => -p_token * (self.size() as f64),
            Prior::SimpleGenerative {
                p_rule,
                atom_weights,
            } => {
                let p_num_rules = |k| fail_geometric_logpdf(k, 1.0 - p_rule);
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
                let p_num_rules = |k| block_generative_logpdf(p_null, 1.0 - p_rule, k, n_blocks);
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
                let p_num_rules = |k| block_generative_logpdf(p_null, 1.0 - p_rule, k, n_blocks);
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
}

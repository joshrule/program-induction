use hypotheses::{Datum, MCMCable, Temperable};
use inference::Control;
use rand::prelude::*;
use std::f64::NEG_INFINITY;
use utilities::FiniteHistory;

pub struct Proposal<T>(pub T, pub f64);

/// An MCMC Chain.
pub struct MCMCChain<'a, H>
where
    H: MCMCable,
{
    pub(crate) current: H,
    data: &'a [Datum<H>],
    pub(crate) temperature: f64,
    maxval: f64,
    pub(crate) samples: usize,
    proposals: usize,
    acceptances: usize,
    steps_since_improvement: usize,
    history: FiniteHistory<u8>,
}

// /// The iterator by which the MCMCChain generates samples.
// pub struct MCMCChainIterator<'a, 'b, H, R>
// where
//     H: MCMCable,
// {
//     chain: &'b mut MCMCChain<'a, H>,
//     ctl: Control,
//     rng: &'b mut R,
// }

impl<'a, H: MCMCable> MCMCChain<'a, H> {
    /// Construct an MCMC chain given an initial hypothesis, a callback to
    /// execute after collecting each sample, and the data to use in evaluating
    /// the proposals.
    pub fn new(mut h0: H, data: &'a [Datum<H>]) -> Self {
        h0.compute_posterior(&data, None);
        // Not calling the callback yet...
        MCMCChain {
            current: h0,
            data,
            temperature: 1.0,
            maxval: NEG_INFINITY,
            samples: 1,
            proposals: 0,
            acceptances: 0,
            steps_since_improvement: 0,
            history: FiniteHistory::new(100),
        }
    }
    /// Change the data available to the chain, and optionally update the
    /// posterior of the current sample.
    pub fn set_data(&mut self, data: &'a [Datum<H>], recompute_posterior: bool) {
        if recompute_posterior {
            self.current.compute_posterior(&data, None);
        }
        self.data = data;
    }
    /// Return a reference to the current sample.
    pub fn current(&self) -> &H {
        &self.current
    }

    /// Return a mutable reference to the current sample.
    pub fn current_mut(&mut self) -> &mut H {
        &mut self.current
    }
    /// Return the best posterior seen since the last restart.
    pub fn maxval(&self) -> f64 {
        self.maxval
    }
    /// Return the best posterior seen since the last restart.
    pub fn samples(&self) -> usize {
        self.samples
    }
    /// Return the chain's acceptance ratio.
    pub fn acceptance_ratio(&self) -> f64 {
        self.history.mean()
    }
    /// Feed the chain a `Control` structure and a source of randomness to begin collecting samples.
    pub fn iter<'b, R: Rng>(
        &'b mut self,
        ctl: Control,
        rng: &'b mut R,
    ) -> MCMCChainIterator<'a, 'b, H, R> {
        MCMCChainIterator {
            chain: self,
            ctl,
            rng,
        }
    }
    pub fn internal_next<'b, R: Rng>(
        &'b mut self,
        ctl: &mut Control,
        rng: &mut R,
    ) -> Option<&'b H> {
        if !ctl.started() {
            println!("# starting ctl");
            ctl.start();
        }
        while ctl.running() {
            // Track top-hypothesis improvements.
            if self.current.bayes_score().posterior > self.maxval {
                // println!("# improving: {}ms", ctl.runtime_ms());
                self.maxval = self.current.bayes_score().posterior;
                self.steps_since_improvement = 0;
            } else {
                // println!("# no improvement: {}ms", ctl.runtime_ms());
                self.steps_since_improvement += 1;
            }

            // Manage restarts.
            if ctl.restart > 0 && self.steps_since_improvement > ctl.restart {
                println!("# restarting (lack of improvement): {}ms", ctl.runtime_ms());
                self.current = self.current.restart(rng);
                self.current.compute_posterior(&self.data, None);

                self.steps_since_improvement = 0; // reset the counter
                self.maxval = self.current.bayes_score().posterior; // and the new max

                self.samples += 1;

                if self.samples >= ctl.burn {
                    return Some(&self.current);
                }

                continue;
            }

            // Propose (and restart if at -infinity).
            let (mut proposal, fb) = if self.current.bayes_score().posterior > NEG_INFINITY {
                self.current.propose(rng)
            } else {
                (self.current.restart(rng), 0.0)
            };

            self.proposals += 1;

            // A lot of proposals are duplicates. If so, save time by not computing
            // the posterior.
            if proposal == self.current {
                proposal.replicate(&self.current);
            } else {
                proposal.compute_posterior(&self.data, None);
            }

            // Use MH acceptance rule, with some fanciness for NaNs.
            let ratio = proposal.at_temperature(self.temperature)
                - self.current.at_temperature(self.temperature)
                - fb;
            if self.current.bayes_score().posterior.is_nan()
                || self.current.bayes_score().posterior == NEG_INFINITY
                || (!proposal.bayes_score().posterior.is_nan()
                    && (ratio >= 0.0 || rng.gen::<f64>() < ratio.exp()))
            {
                self.current = proposal;
                self.history.add(1);
                self.acceptances += 1;
            } else {
                self.history.add(0);
            }

            self.samples += 1;
            if self.samples >= ctl.burn && (ctl.thin == 0 || ctl.done_steps % ctl.thin == 0) {
                return Some(&self.current);
            }
        }
        None
    }
}

impl<'a, H> Temperable for MCMCChain<'a, H>
where
    H: MCMCable,
{
    /// Return the current posterior at temperature `t`.
    fn at_temperature(&self, t: f64) -> f64 {
        self.current.at_temperature(t)
    }
}

// impl<'a, 'b, H, R> Iterator for MCMCChainIterator<'a, 'b, H, R>
// where
//     H: MCMCable,
//     R: Rng,
// {
//     type Item = &'b H;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         self.chain.internal_next(&mut self.ctl, self.rng)
//     }
// }

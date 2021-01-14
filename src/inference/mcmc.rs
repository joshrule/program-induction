use hypotheses::{Datum, MCMCable, Temperable};
use inference::Control;
use rand::prelude::*;
use std::f64::NEG_INFINITY;
use utilities::{FHBool, FiniteHistory};

/// An MCMC Chain.
pub struct MCMCChain<'a, H, F>
where
    H: MCMCable,
{
    current: H,
    callback: Option<&'a mut F>,
    data: &'a [Datum<H>],
    temperature: f64,
    maxval: f64,
    pub(crate) samples: usize,
    proposals: usize,
    acceptances: usize,
    steps_since_improvement: usize,
    history: FiniteHistory<FHBool>,
}

impl<'a, H: MCMCable, F: FnMut(&H)> MCMCChain<'a, H, F> {
    /// Construct an MCMC chain given an initial hypothesis, a callback to
    /// execute after collecting each sample, and the data to use in evaluating
    /// the proposals.
    pub fn new(mut h0: H, callback: Option<&'a mut F>, data: &'a [Datum<H>]) -> Self {
        h0.compute_posterior(&data, None);
        // TODO: calling `callback` here might conflict with `burn`.
        if let Some(cb) = callback {
            (cb)(&h0);
        }
        MCMCChain {
            current: h0,
            data,
            callback,
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

    /// Return the chain's acceptance ratio.
    pub fn acceptance_ratio(&self) -> f64 {
        self.history.mean()
    }
    /// Run the chain without termination conditions.
    pub fn run_forever<R: Rng>(&mut self, mut ctl: Control, rng: &mut R) {
        ctl.steps = 0;
        ctl.runtime = 0;
        self.run(ctl, rng)
    }
    /// Run the chain.
    pub fn run<R: Rng>(&mut self, mut ctl: Control, rng: &mut R) {
        ctl.start();
        while ctl.running() {
            // Track top-hypothesis improvements.
            if self.current.bayes_score().posterior > self.maxval {
                self.maxval = self.current.bayes_score().posterior;
                self.steps_since_improvement = 0;
            } else {
                self.steps_since_improvement += 1;
            }

            // Manage restarts.
            if ctl.restart > 0 && self.steps_since_improvement > ctl.restart {
                self.current = self.current.restart(rng);
                self.current.compute_posterior(&self.data, None);

                self.steps_since_improvement = 0; // reset the counter
                self.maxval = self.current.bayes_score().posterior; // and the new max

                self.samples += 1;

                if self.samples >= ctl.burn {
                    if let Some(cb) = self.callback {
                        (cb)(&self.current);
                    }
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
                self.history.add(FHBool(true));
                self.acceptances += 1;
            } else {
                self.history.add(FHBool(false));
            }

            if self.samples >= ctl.burn && (ctl.thin == 0 || ctl.done_steps % ctl.thin == 0) {
                if let Some(cb) = self.callback {
                    (cb)(&self.current);
                }
            }

            if ctl.print > 0 && self.samples % ctl.print == 0 {
                println!("{}", self.current);
            }

            self.samples += 1;
        }
    }
}

impl<'a, H, F> Temperable for MCMCChain<'a, H, F>
where
    H: MCMCable,
{
    /// Return the current posterior at temperature `t`.
    fn at_temperature(&self, t: f64) -> f64 {
        self.current.at_temperature(t)
    }
}

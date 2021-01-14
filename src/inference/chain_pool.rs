use hypotheses::{Datum, MCMCable};
use inference::{Control, MCMCChain};
use itertools::Itertools;
use rand::prelude::*;

/// A pool of concurrently run `MCMCChain`s.
pub struct ChainPool<'a, H, F>
where
    H: MCMCable,
    F: FnMut(&H),
{
    /// The pool of chains.
    pool: Vec<MCMCChain<'a, H, F>>,
    /// The number many steps to take before switching chains.
    pub steps: usize,
    /// The number of milliseconds to run before switching chains.
    pub runtime: usize,
}

impl<'a, H, F> ChainPool<'a, H, F>
where
    H: MCMCable,
    F: FnMut(&H),
{
    /// Construct an MCMC chain pool.
    pub fn new<R: Rng>(
        mut h0: H,
        callback: Option<F>,
        data: &'a [Datum<H>],
        size: usize,
        all_callback: bool,
        rng: &mut R,
    ) -> Self {
        let pool = (0..size)
            .map(|i| {
                if i == 0 {
                    MCMCChain::new(h0.clone(), callback, data)
                } else if all_callback {
                    MCMCChain::new(h0.restart(rng), callback, data)
                } else {
                    MCMCChain::new(h0.restart(rng), None, data)
                }
            })
            .collect_vec();
        ChainPool {
            pool,
            steps: 0,
            runtime: 0,
        }
    }
    /// Change the data available to the pool, and optionally update the
    /// posterior of the current samples.
    pub fn set_data(&mut self, data: &'a [Datum<H>], recompute_posterior: bool) {
        for chain in self.pool.iter_mut() {
            chain.set_data(data, recompute_posterior)
        }
    }
    /// Run the chain without termination conditions.
    pub fn run_forever<R: Rng>(&mut self, mut ctl: Control, rng: &mut R) {
        ctl.steps = 0;
        ctl.runtime = 0;
        self.run(ctl, rng)
    }
    /// Run the chain.
    pub fn run<R: Rng>(&mut self, mut ctl: Control, rng: &mut R) {
        let mut index = 0;
        let pool_size = self.pool.len();

        assert!(self.pool.len() > 0, "Cannot run an empty ChainPool.");

        while ctl.running() {
            let chain = &mut self.pool[index % pool_size];

            // Store how many samples we did so we can track the total number.
            let old_samples = chain.samples;

            // Run the chain.
            chain.run(
                Control::new(self.steps, self.runtime, ctl.burn, ctl.thin, ctl.restart, 0),
                rng,
            );

            // Update ctl's done_steps (samples taken might have been anything).
            ctl.done_steps += chain.samples - old_samples;

            index += 1;
        }
    }
}

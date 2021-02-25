use hypotheses::{Datum, MCMCable};
use inference::{Control, MCMCChain};
use itertools::Itertools;
use rand::prelude::*;

/// A pool of concurrently run `MCMCChain`s.
pub struct ChainPool<'a, H>
where
    H: MCMCable,
{
    /// The pool of chains.
    pool: Vec<(bool, MCMCChain<'a, H>)>,
    /// The number many steps to take before switching chains.
    pub steps: usize,
    /// The number of milliseconds to run before switching chains.
    pub runtime: usize,
    /// The current chain to run.
    index: usize,
}

impl<'a, H> ChainPool<'a, H>
where
    H: MCMCable,
{
    /// Construct an MCMC chain pool.
    pub fn new<R: Rng>(mut h0: H, data: &'a [Datum<H>], size: usize, rng: &mut R) -> Self {
        let pool = (0..size)
            .map(|i| {
                if i == 0 {
                    (true, MCMCChain::new(h0.clone(), data))
                } else {
                    (true, MCMCChain::new(h0.restart(rng), data))
                }
            })
            .collect_vec();
        ChainPool {
            pool,
            steps: 0,
            runtime: 0,
            index: 0,
        }
    }
    /// Change the data available to the pool, and optionally update the
    /// posterior of the current samples.
    pub fn set_data(&mut self, data: &'a [Datum<H>], recompute_posterior: bool) {
        for (active, chain) in self.pool.iter_mut() {
            if *active {
                chain.set_data(data, recompute_posterior);
            }
        }
    }
    /// Set the temperature.
    pub fn set_temperature(&mut self, new_temperature: f64) {
        for (_, chain) in self.pool.iter_mut() {
            chain.set_temperature(new_temperature)
        }
    }
    /// Return the number of samples collected by each chain.
    pub fn samples(&self) -> Vec<usize> {
        self.pool
            .iter()
            .map(|(_, chain)| chain.samples)
            .collect_vec()
    }
    /// Return the acceptance ratio of each chain.
    pub fn acceptance_ratio(&self) -> Vec<f64> {
        self.pool
            .iter()
            .map(|(_, chain)| chain.acceptance_ratio())
            .collect_vec()
    }
    /// Run the chain.
    pub fn internal_next<'b, R: Rng>(
        &'b mut self,
        ctl: &mut Control,
        rng: &mut R,
    ) -> Option<&'b H> {
        let pool_size = self.pool.len();
        assert!(pool_size > 0, "Cannot run an empty ChainPool.");
        if !ctl.started() {
            ctl.start();
        }
        loop {
            if !ctl.running() {
                return None;
            } else {
                let index = self.index % pool_size;
                match self.pool[index] {
                    (false, _) => {
                        self.index += 1;
                    }
                    (ref mut active, ref mut chain) => {
                        let sample = chain.internal_next(ctl, rng);
                        if sample.is_some() {
                            break;
                        } else {
                            *active = false;
                            self.index += 1;
                        }
                    }
                }
            }
        }
        let current = self.index % pool_size;
        self.index += 1;
        Some(&self.pool[current].1.current)
    }
}

use hypotheses::{Datum, MCMCable, Temperable};
use inference::{Control, MCMCChain};
use itertools::Itertools;
use rand::prelude::*;
use std::time::{Duration, Instant};
use utilities::FiniteHistory;

fn k(t: f64, v: f64, t0: f64) -> f64 {
    (1.0 / v) * t0 / (t + t0)
}

pub struct TemperatureLadder<T>(pub Vec<T>);

/// Specifications for constructing a temperature ladder as used, e.g., in `ParallelTempering`.
pub enum TemperatureLadderSpec {
    /// Specify using the number of temperatures and the highest temperature.
    SizeAndMax(usize, f64),
    /// Specify all but the lowest option (fixed to 1.0) explicitly.
    Explicit(Vec<f64>),
}

/// A pool of concurrently run `MCMCChain`s.
pub struct ParallelTempering<'a, H>
where
    H: MCMCable,
{
    /// The pool of chains.
    pool: Vec<(bool, MCMCChain<'a, H>)>,
    /// The number many steps to take before switching chains.
    pub steps: usize,
    /// The number of milliseconds to run before switching chains.
    pub runtime: usize,
    // /// The number of milliseconds to run before adapting temperatures.
    // pub adapt: usize,
    /// The number of milliseconds to run before proposing swaps.
    pub swap: usize,
    /// A record of whether swaps occurred during past attempts.
    swapped: Vec<FiniteHistory<u8>>,
    /// The current chain to run.
    index: usize,
    /// The last time a swap was proposed.
    last_swap: Instant,
    // /// The last time temperatures were adapted.
    // last_adapt: Instant,
}

impl TemperatureLadderSpec {
    pub fn make(self) -> Vec<f64> {
        match self {
            TemperatureLadderSpec::Explicit(ladder) => ladder,
            TemperatureLadderSpec::SizeAndMax(size, max) => {
                if size == 1 {
                    vec![1.0]
                } else {
                    (0..size)
                        .map(|rung| (rung as f64 * max.ln() / ((size - 1) as f64)).exp())
                        .collect_vec()
                }
            }
        }
    }
}

impl<'a, H> ParallelTempering<'a, H>
where
    H: MCMCable,
{
    pub fn new<R: Rng>(
        mut h0: H,
        data: &'a [Datum<H>],
        ladder: TemperatureLadder<<H as Temperable>::TemperatureSpecification>,
        swap: usize,
        // adapt: usize,
        rng: &mut R,
    ) -> Self {
        let pool = ladder
            .0
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut chain = if i == 0 {
                    MCMCChain::new(h0.clone(), data)
                } else {
                    MCMCChain::new(h0.restart(rng), data)
                };
                chain.set_temperature(*t);
                (true, chain)
            })
            .collect_vec();
        let swapped = (0..pool.len())
            .map(|_| FiniteHistory::new(100))
            .collect_vec();
        ParallelTempering {
            // adapt,
            index: 0,
            pool,
            runtime: 0,
            steps: 0,
            swap,
            swapped,
            // last_adapt: Instant::now(),
            last_swap: Instant::now(),
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
    /// Change the data available to the pool, and optionally update the
    /// posterior of the current samples.
    pub fn set_data(&mut self, data: &'a [Datum<H>], recompute_posterior: bool) {
        for (active, chain) in self.pool.iter_mut() {
            if *active {
                chain.set_data(data, recompute_posterior);
            }
        }
    }
    /// Run the chain.
    pub fn internal_next<'b, R: Rng>(
        &'b mut self,
        ctl: &mut Control,
        rng: &mut R,
    ) -> Option<&'b H> {
        // TODO: This is basically the chain pool code. Refactor so you DRY.
        let pool_size = self.pool.len();

        assert!(pool_size > 0, "Cannot run an empty ChainPool.");

        if !ctl.started() {
            ctl.start();
        }
        loop {
            if !ctl.running() {
                return None;
            } else {
                // Do we need to update the pool?
                self.maybe_swap(rng);
                // self.maybe_adapt();
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
    // fn adapt(&mut self, v: f64, t0: f64) {
    //     let pool_size = self.pool.len();
    //     let mut sw = vec![0f64; pool_size];

    //     for i in 1..pool_size - 1 {
    //         sw[i] = (self.pool[i].1.temperature - self.pool[i - 1].1.temperature).ln();
    //         if self.swapped[i].n() > 0 && self.swapped[i + 1].n() > 0 {
    //             sw[i] += k(self.pool[i].1.samples as f64, v, t0)
    //                 * (self.swapped[i].mean() - self.swapped[i + 1].mean());
    //         }
    //     }

    //     // Convert to temperatures again.
    //     for i in 1..pool_size - 1 {
    //         self.pool[i].1.temperature = self.pool[i - 1].1.temperature + sw[i].exp();
    //     }
    // }
    fn swap<R: Rng>(&mut self, rng: &mut R) {
        // Swap k and k-1.
        let k = 1 + rng.gen_range(0..self.pool.len() - 1);

        // compute R based on data
        let tnow = self.pool[k - 1]
            .1
            .at_temperature(self.pool[k - 1].1.temperature)
            + self.pool[k].1.at_temperature(self.pool[k].1.temperature);
        let tswp = self.pool[k - 1]
            .1
            .at_temperature(self.pool[k].1.temperature)
            + self.pool[k]
                .1
                .at_temperature(self.pool[k - 1].1.temperature);
        let r = tswp - tnow;

        if r >= 0.0 || rng.gen::<f64>() < r.exp() {
            // Swap the chains' hypotheses without angering the borrow checker.
            let (left_slice, right_slice) = self.pool.split_at_mut(k);
            let left = &mut left_slice[k - 1];
            let right = &mut right_slice[0];
            std::mem::swap(&mut left.1.current, &mut right.1.current);
            self.swapped[k].add(1);
        } else {
            self.swapped[k].add(0);
        }
    }
    // fn maybe_adapt(&mut self) {
    //     if self.last_adapt.elapsed() >= Duration::from_millis(self.adapt as u64) {
    //         // TODO: magic constants...
    //         self.adapt(3.0, 1000000.0);
    //         self.last_adapt = Instant::now();
    //     }
    // }
    fn maybe_swap<R: Rng>(&mut self, rng: &mut R) {
        if self.last_swap.elapsed() >= Duration::from_millis(self.swap as u64) {
            self.swap(rng);
            self.last_swap = Instant::now();
        }
    }
}

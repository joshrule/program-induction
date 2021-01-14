//! Search and Inference Algorithms.

// mod chain_pool;
// mod control;
// mod mcmc;
// pub use self::{chain_pool::*, control::Control, mcmc::*};
// use atomic_counter::RelaxedCounter;
// use crossbeam_utils::thread::scope;
//
// pub trait Threadable: Send {
//     fn run_thread(&mut self, counter: &RelaxedCounter, ctl: Control);
// }
//
// pub struct ParallelInference {
//     counter: RelaxedCounter,
// }
//
// impl ParallelInference {
//     pub fn new(initial_index: usize) -> Self {
//         ParallelInference {
//             counter: RelaxedCounter::new(initial_index),
//         }
//     }
//     pub fn run<T: Threadable>(&self, mut chains: T, ctl: Control, n_threads: usize) {
//         scope(|scope| {
//             for _ in 0..n_threads {
//                 scope.spawn(|_| chains.run_thread(&self.counter, ctl));
//             }
//         });
//     }
// }

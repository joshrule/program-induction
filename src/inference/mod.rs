//! Search and Inference Algorithms.

// mod chain_pool;
mod control;
mod mcmc;
// mod parallel_tempering;
pub use self::{
    //chain_pool::*,
    control::Control,
    mcmc::*,
    // parallel_tempering::*,
};

//! Tools for defining hypotheses and hypothesis spaces.

use crate::utilities::f64_eq;
use rand::prelude::*;
use std::{
    f64::{EPSILON, NAN, NEG_INFINITY},
    fmt::Display,
};

/// The data used to evaluate a given `Hypothesis`.
pub type Datum<B> = <B as Bayesable>::Datum;

pub trait Temperable {
    fn at_temperature(&self, t: f64) -> f64;
}

/// `Created` types keep track of when they were created.
pub trait Created {
    type Record: Sized + Copy;
    fn created(&self) -> Self::Record;
}

/// `Hypothesis` types form a hypothesis space which can be searched.
pub trait Hypothesis: Sized + Clone + Eq + Display {}

/// A container for the components of a Bayesian posterior probability.
#[derive(Copy, Clone, Debug)]
pub struct BayesScore {
    pub prior: f64,
    pub likelihood: f64,
    pub posterior: f64,
}

/// `Bayesable` types support Bayesian inference (e.g. computing priors, likelihoods, and posteriors).
pub trait Bayesable: Hypothesis {
    type Datum: Clone + Sized;
    fn bayes_score(&self) -> &BayesScore;
    fn bayes_score_mut(&mut self) -> &mut BayesScore;
    fn compute_prior(&mut self) -> f64;
    fn compute_single_likelihood(&mut self, data: &Self::Datum) -> f64;
    fn compute_likelihood(&mut self, data: &[Self::Datum], breakout: Option<f64>) -> f64 {
        let breakout = breakout.unwrap_or(NEG_INFINITY);
        let mut likelihood = 0.0;
        for datum in data {
            likelihood += self.compute_single_likelihood(datum);

            // Break if the likelihood is garbage.
            if likelihood == NEG_INFINITY || likelihood.is_nan() {
                break;
            };

            // Break if likelihood is too low.
            if likelihood < breakout {
                likelihood = NEG_INFINITY; // should not matter what value, but let's make it -infinity
                break;
            }
        }
        let score = self.bayes_score_mut();
        score.likelihood = likelihood;
        likelihood
    }
    fn compute_posterior(&mut self, data: &[Self::Datum], breakout: Option<f64>) -> f64 {
        let prior = self.compute_prior();
        if (prior - NEG_INFINITY).abs() < EPSILON {
            let mut score = self.bayes_score_mut();
            score.prior = prior;
            score.likelihood = NAN;
            score.posterior = NEG_INFINITY;
            score.posterior
        } else {
            let likelihood = self.compute_likelihood(data, breakout);
            let mut score = self.bayes_score_mut();
            score.prior = prior;
            score.likelihood = likelihood;
            score.posterior = score.prior + score.likelihood;
            score.posterior
        }
    }
}

/// `MCMCable` hypothesis spaces can be searched using MCMC.
pub trait MCMCable: Bayesable {
    fn restart<R: Rng>(&mut self, rng: &mut R) -> Self;
    fn propose<R: Rng>(&mut self, rng: &mut R) -> (Self, f64);
    fn replicate(&mut self, other: &Self);
}

impl<T: Bayesable> Temperable for T {
    fn at_temperature(&self, t: f64) -> f64 {
        let score = self.bayes_score();
        score.prior + score.likelihood / t
    }
}

impl PartialEq for BayesScore {
    fn eq(&self, other: &Self) -> bool {
        f64_eq(self.prior, other.prior)
            && f64_eq(self.likelihood, other.likelihood)
            && f64_eq(self.posterior, other.posterior)
    }
}

impl Eq for BayesScore {}

impl Default for BayesScore {
    fn default() -> Self {
        BayesScore {
            prior: NAN,
            likelihood: NAN,
            posterior: NAN,
        }
    }
}

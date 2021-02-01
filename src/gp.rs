//! Representations capable of Genetic Programming.

use itertools::Itertools;
use polytype::TypeSchema;
use rand::{
    distributions::Distribution,
    distributions::WeightedIndex,
    seq::{IteratorRandom, SliceRandom},
    Rng,
};
use std::cmp::Ordering;

use Task;

/// The mechanism by which individuals are selected for inclusion in the
/// population.
#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub enum GPSelection {
    /// `Deterministic` implies a strict survival-of-the-fittest selection
    /// mechanism, in which the best individuals are always retained. An
    /// individual can only be removed from a population if a better-scoring
    /// individual arises to take its place.
    #[serde(alias = "deterministic")]
    Deterministic,
    /// `Drift(alpha)` implies a noisy survival-of-the-fittest selection
    /// mechanism, in which individuals are selected probabilistically without
    /// replacement from the combination of the population and offspring to form
    /// a new population. Offspring fitness is simply determined by the task,
    /// while the fitness of members of the pre-existing population is computed
    /// as a linear combination of their prior fitness and fitness on the
    /// current task, given as `fitness_{:t} = alpha * fitness_{:t-1} +
    /// (1-alpha) * fitness_t`. An individual can be removed from a population
    /// by lower-scoring individuals, though this is relatively unlikely.
    #[serde(alias = "drift")]
    Drift(f64),
    /// `Hybrid` implies a selection mechanism in which some portion of the
    /// population is selected deterministically such that the best individuals
    /// are always retained. The remainder of the population is sampled without
    /// replacement from the remaining individuals. An individual can be removed
    /// from a population by lower-scoring individuals, though this is
    /// relatively unlikely, and impossible if the individual is considered one
    /// of the "best" in the population. The number of "best" individuals is
    /// `floor(population.len() * deterministic_proportion)`, where
    /// `deterministic_proportion` is `GPSelection::Hybrid.0`. It should vary
    /// from 0 to 1.
    #[serde(alias = "hybrid")]
    Hybrid(f64),
    /// `Probabilistic` implies a noisy survival-of-the-fittest selection
    /// mechanism, in which a population is selected probabilistically from a
    /// set of possible populations in proportion to its overall fitness. An
    /// individual can be removed from a population even by lower-scoring
    /// individuals, though this is relatively unlikely.
    #[serde(alias = "probabilistic")]
    Probabilistic,
    /// `Resample` implies that individuals are selected by sampling from the
    /// offspring *with* replacement, as in a particle filter.
    #[serde(alias = "resample")]
    Resample,
    /// `Sample` implies that individuals are selected by sampling from the
    /// offspring *without* replacement.
    #[serde(alias = "sample")]
    Sample,
}

impl GPSelection {
    pub(crate) fn select<'a, R, X, F>(
        &self,
        population: &mut Vec<(X, f64)>,
        children: Vec<X>,
        oracle: F,
        rng: &mut R,
        pop_size: usize,
    ) where
        R: Rng,
        X: Clone + Send + Sync,
        F: Fn(&X) -> f64 + Send + Sync + 'a,
    {
        let mut scored_children = children
            .into_iter()
            .map(|child| {
                let fitness = oracle(&child);
                (child, fitness)
            })
            .filter(|(_, fitness)| fitness.is_finite())
            .collect_vec();
        match self {
            GPSelection::Sample => {
                population.extend(scored_children);
                *population = sample_without_replacement(population, pop_size, rng);
            }
            GPSelection::Drift(alpha) => {
                for (p, old_fitness) in population.iter_mut() {
                    let new_fitness = oracle(p);
                    *old_fitness = alpha * *old_fitness + (1.0 - alpha) * new_fitness;
                }
                population.extend(scored_children);
                *population = sample_without_replacement(population, pop_size, rng);
            }
            GPSelection::Resample => {
                *population = sample_with_replacement(&mut scored_children, pop_size, rng);
            }
            GPSelection::Deterministic => {
                population.truncate(pop_size);
                while population.len() < pop_size {
                    if let Some(child) = scored_children.pop() {
                        population.push(child);
                    } else {
                        return;
                    }
                }
                for child in scored_children {
                    sorted_place(child, population);
                }
            }
            _ => {
                let mut options = Vec::with_capacity(pop_size + scored_children.len());
                options.append(population);
                options.append(&mut scored_children);
                let mut sample_size = pop_size;
                if let GPSelection::Hybrid(det_proportion) = self {
                    let n_best = options
                        .len()
                        .min((pop_size as f64 * det_proportion).floor() as usize);
                    sample_size -= n_best;
                    options.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                    let rest = options.split_off(n_best);
                    *population = options;
                    options = rest;
                }
                population.append(&mut sample_without_replacement(
                    &mut options,
                    sample_size,
                    rng,
                ));
            }
        }
    }
}

/// Weights for various reproduction operators in genetic programming.
/// Parameters for genetic programming.
#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct GPParams {
    /// The mechanism by which individuals are selected for inclusion in the
    /// population.
    pub selection: GPSelection,
    /// The number of individuals in the population.
    pub population_size: usize,
    /// The number of individuals selected uniformly at random to participate in
    /// a tournament. If 1, a single individual is selected uniformly at random,
    /// as if the population were unweighted. This is useful for mimicking
    /// uniform weights after resampling, as in a particle filter.
    pub tournament_size: usize,
    /// If `true`, weight parent selection for single-individual tournaments.
    pub weighted: bool,
    /// The number of new children considered during each step of evolution.
    /// Traditionally, this would be set to 1. [`reproduce`] is called
    /// repeatedly until producing `n_delta` children.
    ///
    /// [`reproduce`]: #tymethod.reproduce
    pub n_delta: usize,
}

pub struct Tournament<'a, T> {
    pub n: usize,
    pub weighted: bool,
    pub population: &'a [(T, f64)],
}
impl<'a, T> Tournament<'a, T> {
    pub fn new(n: usize, weighted: bool, population: &'a [(T, f64)]) -> Tournament<'a, T> {
        Tournament {
            n,
            weighted,
            population,
        }
    }
}
impl<'a, T> Distribution<&'a T> for Tournament<'a, T> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> &'a T {
        if self.n == 1 {
            if self.population.is_empty() {
                panic!("tournament cannot select winner without contestants");
            } else if self.population.len() == 1 {
                &self.population[0].0
            } else if self.weighted {
                &self
                    .population
                    .choose_weighted(rng, |(_, w)| (-w).exp())
                    .unwrap()
                    .0
            } else {
                &self.population[rng.gen_range(0..self.population.len())].0
            }
        } else {
            (0..self.population.len())
                .choose_multiple(rng, self.n)
                .into_iter()
                .map(|i| &self.population[i])
                .max_by(|&&(_, ref x), &&(_, ref y)| x.partial_cmp(y).expect("found NaN"))
                .map(|&(ref expr, _)| expr)
                .expect("tournament cannot select winner with no contestants")
        }
    }
}

/// A kind of representation suitable for **genetic programming**.
///
/// Implementors of `GP` must provide methods for [`genesis`], [`mutate`], [`crossover`]. A
/// [`Task`] provides a fitness function via its [`oracle`]: we adopt the convention that smaller
/// values are better (so one can think of the [`oracle`] as providing a measure of error). To opt
/// out of the default tournament-based selection, implementors may override the [`tournament`]
/// method.
///
/// Typically, you will interact with this trait using the [`init`] and [`evolve`] methods on
/// existing implementations, such as [`pcfg::Grammar`].
///
/// The provided methods (namely [`init`] and [`evolve`]) manipulate a _population_, which is a
/// sorted list of programs according to fitness. If you otherwise modify the population such that
/// it is no longer sorted appropriately, these methods will behave incorrectly.
///
/// # Examples
///
/// This example has a lot of code, but it is quite straightforward. Given a PCFG and an evaluator
/// for sentences, a task is created by measuring proximity of an evaluted sentence to some target.
/// We finally pick some parameters and evolve a population for a few hundred generations.
///
/// ```
/// #[macro_use] extern crate polytype;
/// extern crate programinduction;
/// extern crate rand;
/// use programinduction::pcfg::{self, Grammar, Rule};
/// use programinduction::{GP, GPParams, GPSelection, Task};
/// use rand::{rngs::SmallRng, SeedableRng};
///
/// fn evaluator(name: &str, inps: &[i32]) -> Result<i32, ()> {
///     match name {
///         "0" => Ok(0),
///         "1" => Ok(1),
///         "plus" => Ok(inps[0] + inps[1]),
///         _ => unreachable!(),
///     }
/// }
///
/// let g = Grammar::new(
///     tp!(EXPR),
///     vec![
///         Rule::new("0", tp!(EXPR), 1.0),
///         Rule::new("1", tp!(EXPR), 1.0),
///         Rule::new("plus", tp!(@arrow[tp!(EXPR), tp!(EXPR), tp!(EXPR)]), 1.0),
///     ],
/// );
/// let target = 6;
/// let task = Task {
///     oracle: Box::new(|g: &Grammar, expr| {
///         if let Ok(n) = g.eval(expr, &evaluator) {
///             (n - target).abs() as f64 // numbers close to target
///         } else {
///             std::f64::INFINITY
///         }
///     }),
///     tp: ptp!(EXPR),
///     observation: (),
/// };
///
/// let gpparams = GPParams {
///     selection: GPSelection::Deterministic,
///     population_size: 10,
///     tournament_size: 5,
///     n_delta: 1,
///     weighted: false,
/// };
/// let params = pcfg::GeneticParams::default();
/// let generations = 1000;
/// let rng = &mut SmallRng::from_seed([1u8; 16]);
///
/// let mut pop = g.init(&params, rng, &gpparams, &task);
/// let mut seen = vec![];
/// for _ in 0..generations {
///     g.evolve(&params, rng, &gpparams, &task, &mut seen, &mut pop)
/// }
///
/// // perfect winner is found!
/// let &(ref winner, score) = &pop[0];
/// assert_eq!(6, g.eval(winner, &evaluator).unwrap());
/// assert_eq!(0.0, score);
/// ```
///
/// [`genesis`]: #tymethod.genesis
/// [`mutate`]: #tymethod.mutate
/// [`crossover`]: #tymethod.crossover
/// [`tournament`]: #method.tournament
/// [`init`]: #method.mutate
/// [`evolve`]: #method.crossover
/// [`Task`]: struct.Task.html
/// [`oracle`]: struct.Task.html#structfield.oracle
/// [`pcfg::Grammar`]: pcfg/struct.Grammar.html
pub trait GP: Send + Sync + Sized {
    /// A Representation is the formalism in which learning takes place, typically `Self`.
    type Representation: Send + Sync + Sized;
    /// An Expression is a sentence in the representation. **Tasks are solved by Expressions**.
    type Expression: Clone + Send + Sync;
    /// Extra parameters for a representation go here.
    type Params;
    // task-specific information, e.g. an input/output pair, goes here.
    type Observation: Clone + Send + Sync;

    /// Create an initial population for a particular requesting type.
    fn genesis<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        pop_size: usize,
        tp: &TypeSchema,
    ) -> Vec<Self::Expression>;

    /// Initializes a population, which is a list of programs and their scores sorted by score.
    /// The most-fit individual is the first element in the population.
    // TODO: uses sneaky magic from https://stackoverflow.com/questions/50090578
    //       that we'll remove after the referenced bugfixes.
    fn init<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        gpparams: &GPParams,
        task: &Task<Self::Representation, Self::Expression, Self::Observation>,
    ) -> Vec<(Self::Expression, f64)>
    where
        for<'a> &'a Self::Representation: From<&'a Self>,
    {
        let exprs = self.genesis(params, rng, gpparams.population_size, &task.tp);
        exprs
            .into_iter()
            .map(|expr| {
                let l = (task.oracle)(<&Self::Representation>::from(self), &expr);
                (expr, l)
            })
            .sorted_by(|&(_, ref x), &(_, ref y)| x.partial_cmp(y).expect("found NaN"))
            .collect()
    }

    /// This should be the guts of `impl GP`. `reproduce` takes parameters,
    /// observations, and a tournament for selecting individuals from the
    /// population, and returns a vector of offspring.
    fn reproduce<R: Rng>(
        &self,
        task: &Task<Self::Representation, Self::Expression, Self::Observation>,
        rng: &mut R,
        params: &Self::Params,
        gpparams: &GPParams,
        obs: &Self::Observation,
        tournament: &Tournament<Self::Expression>,
    ) -> Vec<Self::Expression>;

    /// This should be a filter-like operation on `offspring`. The intended
    /// semantics is that `validate_offspring` reduces the set of newly created
    /// individuals in `offspring` to just those viable for consideration as
    /// part of the total `population`, taking into account other `children`
    /// that are part of the current generation. This allows you to do things
    /// like ensure a population of unique individuals.
    fn validate_offspring(
        &self,
        _params: &Self::Params,
        _population: &[(Self::Expression, f64)],
        _children: &[Self::Expression],
        _seen: &mut Vec<Self::Expression>,
        _offspring: &mut Vec<Self::Expression>,
        _max_validated: usize,
    ) {
    }

    /// Evolves a population. This will repeatedly run a Bernoulli trial with parameter
    /// [`mutation_prob`] and perform mutation or crossover depending on the outcome until
    /// [`n_delta`] expressions are determined.
    ///
    /// [`mutation_prob`]: struct.GPParams.html#mutation_prob
    /// [`n_delta`]: struct.GPParams.html#n_delta
    // TODO: uses sneaky magic from https://stackoverflow.com/questions/50090578
    //       that we'll remove after the referenced bugfixes.
    fn evolve<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        gpparams: &GPParams,
        task: &Task<Self::Representation, Self::Expression, Self::Observation>,
        seen: &mut Vec<Self::Expression>,
        population: &mut Vec<(Self::Expression, f64)>,
    ) where
        for<'a> &'a Self::Representation: From<&'a Self>,
    {
        let mut children = Vec::with_capacity(gpparams.n_delta);
        while children.len() < gpparams.n_delta {
            let tournament =
                Tournament::new(gpparams.tournament_size, gpparams.weighted, population);
            let mut offspring =
                self.reproduce(task, rng, params, gpparams, &task.observation, &tournament);
            let keep_n = gpparams.n_delta - children.len();
            self.validate_offspring(params, population, &children, seen, &mut offspring, keep_n);
            children.append(&mut offspring);
        }
        gpparams.selection.select(
            population,
            children,
            |child| (task.oracle)(<&Self::Representation>::from(self), child),
            rng,
            gpparams.population_size,
        );
    }
}

/// Given a `Vec` of item-score pairs sorted by score, and some `sample_size`,
/// return a score-sorted subset sampled without replacement from the `Vec`
/// according to score.
fn sample_without_replacement<R: Rng, T: Clone>(
    options: &mut Vec<(T, f64)>,
    sample_size: usize,
    rng: &mut R,
) -> Vec<(T, f64)> {
    let sample_size = sample_size.min(options.len());
    if sample_size == 0 {
        return vec![];
    } else if sample_size == options.len() {
        return options.clone();
    }
    let weights = options
        .iter()
        .map(|(_, weight)| (-weight).exp().max(1e-6))
        .collect_vec();
    let mut sample = Vec::with_capacity(sample_size);
    let mut dist = WeightedIndex::new(&weights[..]).unwrap();
    for _ in 0..sample_size {
        let sampled_idx = dist.sample(rng);
        sample.push(options[sampled_idx].clone());
        dist.update_weights(&[(sampled_idx, &0.0)]).ok();
    }
    sample.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    sample
}

/// Given a `Vec` of item-score pairs sorted by score, and some `sample_size`,
/// return a score-sorted subset sampled with replacement from the `Vec`
/// according to score.
fn sample_with_replacement<R: Rng, T: Clone>(
    options: &mut Vec<(T, f64)>,
    sample_size: usize,
    rng: &mut R,
) -> Vec<(T, f64)> {
    let dist = WeightedIndex::new(options.iter().map(|(_, weight)| (-weight).exp())).unwrap();
    let mut sample = Vec::with_capacity(sample_size);
    for _ in 0..sample_size {
        // cloning because we don't know if we'll be using multiple times
        sample.push(options[dist.sample(rng)].clone());
    }
    sample.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    sample
}

/// Given a mutable vector, `pop`, of item-score pairs sorted by score, insert
/// an item, `child`, into `pop` while maintaining score-sorted order.
///
/// The length of the list does *not* change, so if the item would be inserted
/// at the end of the list, no insertion occurs. Also, if existing items have
/// the same score as `child`, `child` is inserted *after* these items.
///
/// This function calls `unsafe` methods but in ways that should not fail.
fn sorted_place<T>(child: (T, f64), pop: &mut Vec<(T, f64)>) {
    let orig_size = pop.len();
    let mut size = orig_size;
    if size == 0 {
        return;
    }
    let idx = {
        let mut base = 0usize;
        while size > 1 {
            let half = size / 2;
            let mid = base + half;
            // mid is always in [0, size), that means mid is >= 0 and < size.
            // mid >= 0: by definition
            // mid < size: mid = size / 2 + size / 4 + size / 8 ...
            let other = unsafe { pop.get_unchecked(mid) };
            let cmp = other.1.partial_cmp(&child.1).expect("found NaN");
            base = if cmp == Ordering::Greater { base } else { mid };
            size -= half;
        }
        // base is always in [0, size) because base <= mid.
        let other = unsafe { pop.get_unchecked(base) };
        let cmp = other.1.partial_cmp(&child.1).expect("found NaN");
        base + (cmp != Ordering::Greater) as usize
    };
    if idx < orig_size {
        pop.pop();
        pop.insert(idx, child);
    }
}

//! (representation) Polymorphically-typed term rewriting system.
//!
//! An evaluatable first-order [Term Rewriting System][0] (TRS) with a [Hindley-Milner type
//! system][1].
//!
//! [0]: https://wikipedia.org/wiki/Hindley–Milner_type_system
//!      "Wikipedia - Hindley-Milner Type System"
//! [1]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//!      "Wikipedia - Term Rewriting Systems"
//!
//! # Example
//!
//! ```
//! # #[macro_use] extern crate polytype;
//! # extern crate programinduction;
//! # extern crate term_rewriting;
//! # use programinduction::trs::{parse_lexicon, parse_trs};
//! # use polytype::atype::with_ctx;
//! with_ctx(32, |ctx| {
//!     let mut lex = parse_lexicon(
//!         "PLUS/2: int -> int -> int; SUCC/1: int-> int; ZERO/0: int;",
//!         &ctx,
//!     ).expect("lex");
//!
//!     let trs = parse_trs(
//!         "PLUS(v0_ ZERO) = v0_; PLUS(v0_ SUCC(v1_)) = SUCC(PLUS(v0_ v1_));",
//!         &mut lex, true, &[]
//!     ).expect("trs");
//!
//!     assert_eq!(trs.len(), 2);
//! })
//! ```

// TODO: reinstate
// pub mod gp;
mod environment;
mod lexicon;
pub mod mcts;
pub mod parser;
mod rewrite;

pub use self::environment::{AtomEnumeration, Env, SampleParams};
pub use self::lexicon::{GenerationLimit, Lexicon};
pub use self::parser::{
    parse_context, parse_lexicon, parse_rule, parse_rulecontext, parse_rulecontexts, parse_rules,
    parse_term, parse_trs,
};
pub use self::rewrite::{Composition, Recursion, Variablization, TRS};
use polytype::atype::UnificationError;
use std::fmt;
use term_rewriting::{PStringDist, Rule, Strategy as RewriteStrategy, TRSError, Term};

#[derive(Debug, Clone)]
/// The error type for type inference.
pub enum TypeError<'ctx> {
    Unification(UnificationError<'ctx>),
    NotFound,
    Malformed,
}
impl<'ctx> From<UnificationError<'ctx>> for TypeError<'ctx> {
    fn from(e: UnificationError<'ctx>) -> Self {
        TypeError::Unification(e)
    }
}
impl<'ctx> fmt::Display for TypeError<'ctx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TypeError::Unification(ref e) => write!(f, "unification error: {}", e),
            TypeError::NotFound => write!(f, "object not found"),
            TypeError::Malformed => write!(f, "query is nonsense"),
        }
    }
}
impl<'ctx> ::std::error::Error for TypeError<'ctx> {
    fn description(&self) -> &'static str {
        "type error"
    }
}

#[derive(Debug, Clone)]
/// The error type for sampling operations.
pub enum SampleError<'ctx> {
    TypeError(TypeError<'ctx>),
    TRSError(TRSError),
    SizeExceeded,
    OptionsExhausted,
    Subterm,
    Trivial,
}
impl<'ctx> From<TypeError<'ctx>> for SampleError<'ctx> {
    fn from(e: TypeError<'ctx>) -> Self {
        SampleError::TypeError(e)
    }
}
impl<'ctx> From<TRSError> for SampleError<'ctx> {
    fn from(e: TRSError) -> Self {
        SampleError::TRSError(e)
    }
}
impl<'ctx> From<UnificationError<'ctx>> for SampleError<'ctx> {
    fn from(e: UnificationError<'ctx>) -> Self {
        SampleError::TypeError(TypeError::Unification(e))
    }
}
impl<'ctx> fmt::Display for SampleError<'ctx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SampleError::TypeError(ref e) => write!(f, "type error: {}", e),
            SampleError::TRSError(ref e) => write!(f, "TRS error: {}", e),
            SampleError::SizeExceeded => write!(f, "exceeded maximum size"),
            SampleError::OptionsExhausted => write!(f, "failed to sample (options exhausted)"),
            SampleError::Subterm => write!(f, "cannot sample subterm"),
            SampleError::Trivial => write!(f, "result is a trivial rule"),
        }
    }
}
impl<'ctx> std::error::Error for SampleError<'ctx> {
    fn description(&self) -> &'static str {
        "sample error"
    }
}

/// Parameters for a TRS-based probabilistic model.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct ModelParams {
    // The temperature constant.
    pub schedule: Schedule,
    pub prior: Prior,
    pub likelihood: Likelihood,
    /// The weight of the log likelihood in the posterior.
    pub l_temp: f64,
    /// The weight of the prior in the posterior.
    pub p_temp: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Datum {
    Full(Rule),
    Partial(Term),
}

pub enum Eval {
    Full(f64),
    Partial(f64, bool),
}

impl Datum {
    pub fn is_full(&self) -> bool {
        matches!(self, Datum::Full(_))
    }
    pub fn is_partial(&self) -> bool {
        matches!(self, Datum::Partial(_))
    }
}

impl Eval {
    pub fn likelihood(&self) -> f64 {
        match *self {
            Eval::Full(x) => x,
            Eval::Partial(x, _) => x,
        }
    }
    pub fn generalizes(&self) -> bool {
        match *self {
            Eval::Full(_) => true,
            Eval::Partial(_, x) => x,
        }
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub enum Schedule {
    // Don't anneal at all.
    #[serde(alias = "none")]
    None,
    // Don't anneal - just use this temperature.
    #[serde(alias = "constant")]
    Constant(f64),
    // Use a linear cooling schedule.
    #[serde(alias = "linear")]
    Linear(f64, f64),
    // Use a logarithmic cooling schedule.
    #[serde(alias = "logarithmic")]
    Logarithmic(f64, f64),
    // Use an exponential cooling schedule.
    #[serde(alias = "exponential")]
    Exponential(f64, f64),
    // Use an inverse cooling schedule.
    #[serde(alias = "inverse")]
    Inverse(f64, f64),
}

impl Schedule {
    pub fn temperature(&self, t: f64) -> f64 {
        match *self {
            Schedule::None => 1.0,
            Schedule::Constant(c) => c,
            Schedule::Linear(t0, eta) => t0 - eta * t,
            Schedule::Logarithmic(t0, alpha) => t0 / (1.0 + alpha * (1.0 + t).ln()),
            Schedule::Exponential(t0, alpha) => t0 * alpha.powf(t),
            Schedule::Inverse(t0, eta) => t0 / (1.0 + eta * t),
        }
    }
}

pub trait ProbabilisticModel {
    type Object;
    type Datum;
    fn log_prior(&mut self, object: &Self::Object) -> f64;
    fn single_log_likelihood<DataIter>(&mut self, object: &Self::Object, data: &Self::Datum)
        -> f64;
    fn log_likelihood(&mut self, object: &Self::Object, data: &[Self::Datum]) -> f64;
    /// Computes the log posterior of `object` given `data` and returns
    /// `(<log_prior>, <log_likelihood>, <log_posterior>)`.
    fn log_posterior(&mut self, object: &Self::Object, data: &[Self::Datum]) -> (f64, f64, f64) {
        let lprior = self.log_prior(object);
        let llikelihood = self.log_likelihood(object, data);
        let lposterior = lprior + llikelihood;
        (lprior, llikelihood, lposterior)
    }
}

pub struct Hypothesis<O, D, M>
where
    M: ProbabilisticModel<Object = O, Datum = D>,
{
    pub object: O,
    pub model: M,
    pub lprior: f64,
    pub llikelihood: f64,
    pub lposterior: f64,
}

impl<O, D, M> Hypothesis<O, D, M>
where
    M: ProbabilisticModel<Object = O, Datum = D>,
{
    pub fn new(object: O, model: M) -> Self {
        Hypothesis {
            object,
            model,
            lprior: std::f64::NAN,
            llikelihood: std::f64::NAN,
            lposterior: std::f64::NAN,
        }
    }
    pub fn log_posterior(&mut self, data: &[D]) -> f64 {
        let (lprior, llikelihood, lposterior) = self.model.log_posterior(&self.object, data);
        self.lprior = lprior;
        self.llikelihood = llikelihood;
        self.lposterior = lposterior;
        self.lposterior
    }
}

/// Possible priors for a TRS-based probabilistic model.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Prior {
    // size-based prior with a fixed cost per atom
    Size(f64),
    // Generative prior based on sampling from the Lexicon
    SimpleGenerative {
        p_rule: f64,
        atom_weights: (f64, f64, f64, f64),
    },
    BlockGenerative {
        p_rule: f64,
        p_null: f64,
        n_blocks: usize,
        atom_weights: (f64, f64, f64, f64),
    },
    StringBlockGenerative {
        p_rule: f64,
        p_null: f64,
        n_blocks: usize,
        atom_weights: (f64, f64, f64, f64),
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
    },
}

/// Likelihood for a TRS-based probabilistic model.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Likelihood {
    /// The weight of example i, i < n, in the likelihood for trial n = decay^(n - 1 - i).
    pub decay: f64,
    /// The rewriting strategy used in computing the likelihood.
    pub strategy: RewriteStrategy,
    /// The (non-log) probability of generating observations at arbitrary
    /// evaluation steps (i.e. not just normal forms). Typically 0.0.
    pub p_observe: f64,
    /// The number of evaluation steps you would like to explore in the trace. `None` evaluates the entire trace, which may not terminate.
    pub max_steps: usize,
    /// The largest term considered for evaluation. `None` considers all terms.
    pub max_size: Option<usize>,
    /// The deepest level of the `Trace` considered for evaluation. `None`
    /// considers all depths.
    pub max_depth: Option<usize>,
    /// Additional parameters for the function used to compute the likelihood of a single datum.
    pub single: SingleLikelihood,
}

/// Possible single likelihoods for a TRS-based probabilistic model.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum SingleLikelihood {
    // Binary log-likelihood: 0 or -\infty
    Binary,
    // Rational Rules (Goodman, et al., 2008) log-likelihood: 0 or -p_outlier
    Rational(f64),
    // trace-based log-likelihood without noise model: 1-p_trace(h,d)
    Trace,
    // generative trace-based log-likelihood with string edit distance noise model: (1-p_edit(h,d))
    String {
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
    },
    // generative trace-based log-likelihood with string edit distance noise model: (1-p_edit(h,d))
    List {
        alpha: f64,
        atom_weights: (f64, f64, f64, f64),
        dist: PStringDist,
        t_max: usize,
        d_max: usize,
    },
    // generative trace-based log-likelihood with prefix noise model: (1-p_prefix(d|h))
    ListPrefix {
        alpha: f64,
        p_add: f64,
        p_del: f64,
    },
    // Reward generalization.
    Generalization(f64),
}

// TODO: FIXME - I'm broken because TRSs are no longer thread safe.
///// Construct a [`Task`] evaluating [`TRS`]s (constructed from a [`Lexicon`])
///// using rewriting of inputs to outputs.
/////
///// Each [`term_rewriting::Rule`] in `data` must have a single RHS term. The
///// resulting [`Task`] checks whether each datum's LHS gets rewritten to its RHS
///// under a [`TRS`] within the constraints specified by the [`ModelParams`].
/////
///// [`Lexicon`]: struct.Lexicon.html
///// [`ModelParams`]: struct.ModelParams.html
///// [`term_rewriting::Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
///// [`Task`]: ../struct.Task.html
///// [`TRS`]: struct.TRS.html
//pub fn task_by_rewrite<'a, 'b, 'c, O: Sync>(
//    data: &'a [Datum],
//    params: ModelParams,
//    lex: &Lexicon<'b, 'c>,
//    t: f64,
//    observation: O,
//) -> Result<Task<'a, Lexicon<'b, 'c>, TRS<'b, 'c>, O>, TypeError<'b>> {
//    Ok(Task {
//        oracle: Box::new(move |_s: &Lexicon, h: &TRS| {
//            -h.log_posterior(data, &mut HashMap::new(), t, params)
//        }),
//        tp: lex.infer_data(data)?,
//        observation,
//    })
//}

fn as_result<'ctx, T>(xs: Vec<T>) -> Result<Vec<T>, SampleError<'ctx>> {
    if xs.is_empty() {
        Err(SampleError::OptionsExhausted)
    } else {
        Ok(xs)
    }
}

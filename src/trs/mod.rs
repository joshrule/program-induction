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
//! # use polytype::Context as TypeContext;
//! let mut lex = parse_lexicon(
//!     "PLUS/2: int -> int -> int; SUCC/1: int-> int; ZERO/0: int;",
//!     TypeContext::default(),
//! )
//!     .expect("parsed lexicon");
//!
//! let trs = parse_trs("PLUS(v0_ ZERO) = v0_; PLUS(v0_ SUCC(v1_)) = SUCC(PLUS(v0_ v1_));", &mut lex, true, &[]);
//! ```

pub mod gp;
mod lexicon;
pub mod mcts;
pub mod parser;
mod rewrite;
pub use self::lexicon::{Environment, GenerationLimit, Lexicon};
pub use self::parser::{
    parse_context, parse_lexicon, parse_rule, parse_rulecontext, parse_rulecontexts, parse_rules,
    parse_term, parse_trs,
};
pub use self::rewrite::{Composition, Recursion, Variablization, TRS};
use Task;

use polytype;
use std::{collections::HashMap, fmt};
use term_rewriting::{PStringDist, Rule, Strategy as RewriteStrategy, TRSError, Term};

#[derive(Debug, Clone)]
/// The error type for type inference.
pub enum TypeError {
    Unification(polytype::UnificationError),
    NotFound,
    Malformed,
}
impl From<polytype::UnificationError> for TypeError {
    fn from(e: polytype::UnificationError) -> TypeError {
        TypeError::Unification(e)
    }
}
impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TypeError::Unification(ref e) => write!(f, "unification error: {}", e),
            TypeError::NotFound => write!(f, "object not found"),
            TypeError::Malformed => write!(f, "query is nonsense"),
        }
    }
}
impl ::std::error::Error for TypeError {
    fn description(&self) -> &'static str {
        "type error"
    }
}

#[derive(Debug, Clone)]
/// The error type for sampling operations.
pub enum SampleError {
    TypeError(TypeError),
    TRSError(TRSError),
    SizeExceeded,
    OptionsExhausted,
    Subterm,
    Trivial,
}
impl From<TypeError> for SampleError {
    fn from(e: TypeError) -> SampleError {
        SampleError::TypeError(e)
    }
}
impl From<TRSError> for SampleError {
    fn from(e: TRSError) -> SampleError {
        SampleError::TRSError(e)
    }
}
impl From<polytype::UnificationError> for SampleError {
    fn from(e: polytype::UnificationError) -> SampleError {
        SampleError::TypeError(TypeError::Unification(e))
    }
}
impl fmt::Display for SampleError {
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
impl ::std::error::Error for SampleError {
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

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Schedule {
    // Don't anneal at all.
    #[serde(alias = "none")]
    None,
    // Don't anneal - just use this temperature.
    #[serde(alias = "constant")]
    Constant(f64),
    // Use a logarithmic cooling schedule.
    #[serde(alias = "logarithmic")]
    Logarithmic(f64),
}

impl Schedule {
    pub fn temperature(&self, t: f64) -> f64 {
        match *self {
            Schedule::None => 1.0,
            Schedule::Constant(c) => c,
            Schedule::Logarithmic(c) => c / (1.0 + t).ln(),
        }
    }
}

pub struct Hypothesis<'a, 'b> {
    pub trs: TRS<'a, 'b>,
    pub lprior: f64,
    pub llikelihood: f64,
    pub lposterior: f64,
    evals: HashMap<Rule, f64>,
    temperature: f64,
    params: ModelParams,
}

impl<'a, 'b> Hypothesis<'a, 'b> {
    pub fn new(
        trs: TRS<'a, 'b>,
        data: &[Rule],
        input: Option<&Term>,
        t: f64,
        params: ModelParams,
    ) -> Hypothesis<'a, 'b> {
        let lprior = trs.log_prior(params.prior);
        let mut evals = HashMap::with_capacity(data.len());
        let llikelihood = trs.log_likelihood(data, input, &mut evals, params.likelihood);
        let temperature = params.schedule.temperature(t);
        let lposterior = (params.p_temp * lprior + params.l_temp * llikelihood) / temperature;
        Hypothesis {
            trs,
            llikelihood,
            lprior,
            lposterior,
            temperature,
            params,
            evals,
        }
    }
    pub fn change_data(&mut self, data: &[Rule], input: Option<&Term>) {
        self.llikelihood =
            self.trs
                .log_likelihood(data, input, &mut self.evals, self.params.likelihood);
        self.lposterior = (self.params.p_temp * self.lprior
            + self.params.l_temp * self.llikelihood)
            / self.temperature;
    }
    pub fn change_time(&mut self, t: f64) {
        self.temperature = self.params.schedule.temperature(t);
        self.lposterior = (self.params.p_temp * self.lprior
            + self.params.l_temp * self.llikelihood)
            / self.temperature;
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
    single: SingleLikelihood,
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
}

/// Construct a [`Task`] evaluating [`TRS`]s (constructed from a [`Lexicon`])
/// using rewriting of inputs to outputs.
///
/// Each [`term_rewriting::Rule`] in `data` must have a single RHS term. The
/// resulting [`Task`] checks whether each datum's LHS gets rewritten to its RHS
/// under a [`TRS`] within the constraints specified by the [`ModelParams`].
///
/// [`Lexicon`]: struct.Lexicon.html
/// [`ModelParams`]: struct.ModelParams.html
/// [`term_rewriting::Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
/// [`Task`]: ../struct.Task.html
/// [`TRS`]: struct.TRS.html
pub fn task_by_rewrite<'a, 'b, 'c, O: Sync>(
    data: &'a [Rule],
    input: Option<&'a Term>,
    params: ModelParams,
    lex: &Lexicon,
    t: f64,
    observation: O,
) -> Result<Task<'a, Lexicon<'c>, TRS<'b, 'c>, O>, TypeError> {
    Ok(Task {
        oracle: Box::new(move |_s: &Lexicon, h: &TRS| {
            -h.log_posterior(data, input, &mut HashMap::new(), t, params)
        }),
        // assuming the data have no variables, we can use the Lexicon's ctx.
        tp: lex.infer_rules(data, &mut lex.0.ctx.clone())?,
        observation,
    })
}

fn as_result<T>(xs: Vec<T>) -> Result<Vec<T>, SampleError> {
    if xs.is_empty() {
        Err(SampleError::OptionsExhausted)
    } else {
        Ok(xs)
    }
}

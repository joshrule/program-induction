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
//! # use programinduction::trs::{TRS, Lexicon};
//! # use polytype::Context as TypeContext;
//! # use term_rewriting::{Signature, parse_rule};
//! let mut sig = Signature::default();
//!
//! let mut ops = vec![];
//! sig.new_op(2, Some("PLUS".to_string()));
//! ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
//! sig.new_op(1, Some("SUCC".to_string()));
//! ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
//! sig.new_op(0, Some("ZERO".to_string()));
//! ops.push(ptp![int]);
//!
//! let rules = vec![
//!     parse_rule(&mut sig, "PLUS(x_ ZERO) = x_").expect("parsed rule"),
//!     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))").expect("parsed rule"),
//! ];
//!
//! let vars = vec![
//!     ptp![int],
//!     ptp![int],
//!     ptp![int],
//! ];
//!
//! let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], vec![], false, TypeContext::default());
//!
//! let trs = TRS::new(&lexicon, rules);
//! ```

mod lexicon;
pub mod parser;
mod rewrite;
pub use self::lexicon::{GPLexicon, GeneticParams, Lexicon};
pub use self::parser::{
    parse_context, parse_lexicon, parse_rule, parse_rulecontext, parse_templates, parse_term,
    parse_trs,
};
pub use self::rewrite::{TRSMove, TRSMoveName, TRSMoves, TRS};
use Task;

use polytype;
use std::fmt;
use term_rewriting::{PStringDist, Rule, Strategy as RewriteStrategy, TRSError};

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
    SizeExceeded(usize, usize),
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
            SampleError::SizeExceeded(size, max_size) => {
                write!(f, "size {} exceeded maximum of {}", size, max_size)
            }
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
    pub prior: Prior,
    pub likelihood: Likelihood,
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
    /// The weight of the log likelihood in the posterior.
    pub l_temp: f64,
    /// The weight of the prior in the posterior.
    pub p_temp: f64,
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

/// Possible likelihoods for a TRS-based probabilistic model.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Likelihood {
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
pub fn task_by_rewrite<'a, 'b, O: Sync>(
    data: &'a [Rule],
    params: ModelParams,
    lex: &Lexicon,
    observation: O,
) -> Result<Task<'a, Lexicon, TRS<'b>, O>, TypeError> {
    Ok(Task {
        oracle: Box::new(move |_s: &Lexicon, h: &TRS| -h.log_posterior(data, params)),
        // assuming the data have no variables, we can use the Lexicon's ctx.
        tp: lex.infer_rules(data)?,
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

//! Tools for making everything else easier.

mod finite_history;

pub use self::finite_history::FiniteHistory;
use itertools::Itertools;
use std::{
    f64::{EPSILON, INFINITY, NEG_INFINITY},
    iter::repeat,
};

#[macro_export]
macro_rules! r#tryo {
    ($state:ident, $expr:expr) => {
        match $expr {
            std::option::Option::Some(val) => val,
            std::option::Option::None => {
                $state.label = StateLabel::Failed;
                return;
            }
        }
    };
}

pub fn f64_eq(f1: f64, f2: f64) -> bool {
    matches!(f1.partial_cmp(&f2), Some(std::cmp::Ordering::Equal) | None)
}

pub fn logsumexp(lps: &[f64]) -> f64 {
    let largest = lps.iter().fold(NEG_INFINITY, |acc, lp| acc.max(*lp));
    if largest == NEG_INFINITY {
        NEG_INFINITY
    } else {
        let x = lps.iter().map(|lp| (lp - largest).exp()).sum::<f64>().ln();
        largest + x
    }
}

pub fn logdiffexp(x: f64, y: f64) -> f64 {
    if (x - y).abs() < EPSILON {
        NEG_INFINITY
    } else {
        let largest = x.max(y);
        let xprime = x - largest;
        let yprime = y - largest;
        largest + (xprime.exp() - yprime.exp()).ln()
    }
}

pub fn exp_normalize(lps: &[f64], rescale: Option<f64>) -> Option<Vec<f64>> {
    let (non_inf_min, max) = lps.iter().fold((INFINITY, NEG_INFINITY), |acc, lp| {
        if *lp == NEG_INFINITY {
            (acc.0, acc.1.max(*lp))
        } else {
            (acc.0.min(*lp), acc.1.max(*lp))
        }
    });
    if max == NEG_INFINITY {
        None
    } else {
        let mut ps = match rescale {
            Some(ceiling) if non_inf_min < ceiling => {
                let factor = non_inf_min / ceiling;
                lps.iter().map(|x| ((x - max) / factor).exp()).collect_vec()
            }
            _ => lps.iter().map(|x| (x - max).exp()).collect_vec(),
        };
        let sum = ps.iter().sum::<f64>();
        let ps_len = ps.len() as f64;
        for p in ps.iter_mut() {
            if sum == 0_f64 {
                *p = 1.0 / ps_len;
            } else {
                *p /= sum;
            }
        }
        Some(ps)
    }
}

pub(crate) fn fail_geometric_logpdf(k: usize, p: f64) -> f64 {
    (1.0 - p).ln() * (k as f64) + p.ln()
}

pub(crate) fn trials_geometric_logpdf(k: usize, p: f64) -> f64 {
    (1.0 - p).ln() * ((k as f64) - 1.0) + p.ln()
}

pub(crate) fn zero_or_trials_geometric_logpdf(k: usize, a: f64, p: f64) -> f64 {
    if k == 0 {
        a.ln()
    } else {
        (1.0 - a).ln() + trials_geometric_logpdf(k, p)
    }
}

pub(crate) fn assignments(n_items: usize, n_groups: usize) -> Option<Vec<Vec<usize>>> {
    if n_items == 0 || n_groups == 0 {
        None
    } else {
        Some(
            repeat(0..n_groups)
                .take(n_items)
                .multi_cartesian_product()
                .collect_vec(),
        )
    }
}

pub(crate) fn assignment_to_count(assignment: &[usize], n_groups: usize) -> Vec<usize> {
    let mut count = vec![0usize; n_groups];
    for a in assignment {
        count[*a] += 1;
    }
    count
}

// - p(0,a,p,m) = a^m
// - p(n,a,p,m) = sum_a prod_i p(a_i,a,p,1), a \in assignments(n,m) , 0 <= i <= m
pub(crate) fn block_generative_logpdf(a: f64, p: f64, n_items: usize, n_blocks: usize) -> f64 {
    if let Some(assignments) = assignments(n_items, n_blocks) {
        let counts = assignments
            .into_iter()
            .map(|x| assignment_to_count(&x[..], n_blocks))
            .collect_vec();
        let lps = counts
            .into_iter()
            .map(|cs| {
                cs.into_iter()
                    .map(|c| zero_or_trials_geometric_logpdf(c, a, p))
                    .sum()
            })
            .collect_vec();
        logsumexp(&lps)
    } else {
        a.ln() * (n_blocks as f64)
    }
}

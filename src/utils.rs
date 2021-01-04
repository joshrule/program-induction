use itertools::Itertools;
use rand::prelude::*;
use std::{cmp::Ordering, f64, iter::repeat};

pub fn logsumexp(lps: &[f64]) -> f64 {
    let largest = lps.iter().fold(f64::NEG_INFINITY, |acc, lp| acc.max(*lp));
    if largest == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else {
        let x = lps.iter().map(|lp| (lp - largest).exp()).sum::<f64>().ln();
        largest + x
    }
}

pub fn logdiffexp(x: f64, y: f64) -> f64 {
    if x == y {
        f64::NEG_INFINITY
    } else {
        let largest = x.max(y);
        let xprime = x - largest;
        let yprime = y - largest;
        largest + (xprime.exp() - yprime.exp()).ln()
    }
}

pub fn exp_normalize(lps: &[f64], rescale: Option<f64>) -> Option<Vec<f64>> {
    let (non_inf_min, max) = lps
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, lp| {
            if *lp == f64::NEG_INFINITY {
                (acc.0, acc.1.max(*lp))
            } else {
                (acc.0.min(*lp), acc.1.max(*lp))
            }
        });
    if max == f64::NEG_INFINITY {
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

/// Randomly permute a mutable slice, `xs`, given a set of weights, `ws`.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate polytype;
/// # extern crate programinduction;
/// # extern crate rand;
/// # use programinduction::weighted_permutation;
/// # use rand::prelude::*;
/// let mut rng = thread_rng();
///
/// let ws = [1.0, 1.0, 1.0, 1.0, 1.0];
/// let mut xs = [5, 6, 7, 8, 9];
/// weighted_permutation(&mut xs, &ws, &mut rng);
/// ```
pub fn weighted_permutation<T: Eq + Clone + std::fmt::Debug, R: Rng>(
    xs: &mut [T],
    ws: &[f64],
    rng: &mut R,
) {
    // Gives a list of the new index and the current index.
    // This is probably cheap to create compared to cloning the Ts.
    let mut indices = ws
        .iter()
        .map(|w| -rng.gen::<f64>().ln() / w)
        .enumerate()
        .sorted_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
        .rev()
        .map(|(i, _)| i)
        .enumerate()
        .rev()
        .collect_vec();
    // The idea here is to keep track of where items are by mutating the index tracker.
    while let Some((new, old)) = indices.pop() {
        if new != old {
            xs.swap(new, old);
            for x in indices.iter_mut() {
                if x.1 == new {
                    x.1 = old;
                }
            }
        }
    }
}

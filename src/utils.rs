use itertools::Itertools;
use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};
use std::f64;
use std::iter::repeat;

#[inline(always)]
pub fn logsumexp(lps: &[f64]) -> f64 {
    let largest = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x = lps.iter().map(|lp| (lp - largest).exp()).sum::<f64>().ln();
    largest + x
}

pub fn fail_geometric_logpdf(k: usize, p: f64) -> f64 {
    (1.0 - p).ln() * (k as f64) + p.ln()
}

pub fn trials_geometric_logpdf(k: usize, p: f64) -> f64 {
    (1.0 - p).ln() * ((k as f64) - 1.0) + p.ln()
}

pub fn zero_or_trials_geometric_logpdf(k: usize, a: f64, p: f64) -> f64 {
    if k == 0 {
        a.ln()
    } else {
        (1.0 - a).ln() + trials_geometric_logpdf(k, p)
    }
}

pub fn assignments(n_items: usize, n_groups: usize) -> Option<Vec<Vec<usize>>> {
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

pub fn assignment_to_count(assignment: &[usize], n_groups: usize) -> Vec<usize> {
    let mut count = vec![0usize; n_groups];
    for a in assignment {
        count[*a] += 1;
    }
    count
}

// - p(0,a,p,m) = a^m
// - p(n,a,p,m) = sum_a prod_i p(a_i,a,p,1), a \in assignments(n,m) , 0 <= i <= m
pub fn block_generative_logpdf(a: f64, p: f64, n_items: usize, n_blocks: usize) -> f64 {
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

// pub fn weighted_permutation<T: Clone>(xs: &[T], ws: &[f64], n: Option<usize>) -> Vec<T> {
//     let mut ws = ws.to_vec();
//     let mut idxs: Vec<_> = (0..(ws.len())).collect();
//     let mut permutation = vec![];
//     let length = cmp::min(n.unwrap_or_else(|| xs.len()), xs.len());
//     while permutation.len() < length {
//         let jidxs: Vec<_> = idxs.iter().cloned().enumerate().collect();
//         let &(jdx, idx): &(usize, usize) = weighted_sample(&jidxs, &ws);
//         permutation.push(xs[idx].clone());
//         idxs.remove(jdx);
//         ws.remove(jdx);
//     }
//     permutation
// }

/// Samples an item from `xs` given the weights `ws`.
pub fn weighted_sample<'a, T>(xs: &'a [T], ws: &[f64]) -> &'a T {
    assert_eq!(xs.len(), ws.len(), "weighted sample given invalid inputs");
    let total = ws.iter().fold(0f64, |acc, x| acc + x);
    let threshold: f64 = Uniform::new(0f64, total).sample(&mut thread_rng());
    let mut cum = 0f64;
    for (wp, x) in ws.iter().zip(xs) {
        cum += *wp;
        if threshold <= cum {
            return x;
        }
    }
    unreachable!()
}

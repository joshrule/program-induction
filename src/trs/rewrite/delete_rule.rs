use itertools::Itertools;
use rand::Rng;
use std::collections::HashMap;
use term_rewriting::{Rule, Term};
use trs::{as_result, SampleError, TRS};

impl<'ctx, 'b> TRS<'ctx, 'b> {
    /// Delete a learned rule from the rewrite system.
    pub fn delete_rule(&self) -> Result<Vec<TRS<'ctx, 'b>>, SampleError<'ctx>> {
        let rules = self
            .utrs
            .rules
            .iter()
            .flat_map(|r| r.clauses())
            .collect_vec();
        if rules.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            rules
                .iter()
                .map(|d| {
                    let mut trs = self.clone();
                    trs.utrs.remove_clauses(d)?;
                    Ok(trs)
                })
                .collect()
        }
    }

    /// Delete all combinations of learned rules from the rewrite system.
    pub fn delete_rules<R: Rng>(
        &self,
        rng: &mut R,
        threshold: usize,
    ) -> Result<Vec<TRS<'ctx, 'b>>, SampleError<'ctx>> {
        let deletable = as_result(self.clauses())?;
        let mut trss = vec![];
        if 2usize.pow(deletable.len() as u32) > 2 * threshold {
            while trss.len() < threshold {
                let mut trs = self.clone();
                // Flip a coin for each clause and remove the successes.
                for (_, rule) in &deletable {
                    if rng.gen() {
                        trs.utrs.remove_clauses(rule)?;
                    }
                }
                if !trss.contains(&trs) {
                    trss.push(trs);
                }
            }
        } else {
            // Exhaustively try all non-trivial deletions.
            for n in 1..deletable.len() {
                for rules in deletable.iter().combinations(n) {
                    let mut trs = self.clone();
                    for (_, rule) in &rules {
                        trs.utrs.remove_clauses(rule)?;
                    }
                    trss.push(trs);
                }
            }
        }
        as_result(trss)
    }

    /// Delete rules from the rewrite system whose LHS matches a prior rule's.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate term_rewriting;
    /// # extern crate rand;
    /// # use programinduction::trs::{parse_lexicon, parse_trs};
    /// # use rand::thread_rng;
    /// # use term_rewriting::Signature;
    /// # use polytype::{Source, atype::{with_ctx, TypeSchema, TypeContext}};
    /// with_ctx(10, |ctx: TypeContext<'_>| {
    ///     let mut rng = thread_rng();
    ///     let mut lex = parse_lexicon(
    ///         "A/2: term -> term -> term; B/1: term -> term; C/0: term;",
    ///         &ctx,
    ///     ).expect("lex");
    ///
    ///     let trs1 = parse_trs(
    ///         "A(v0_ v1_) = v0_; A(v0_ C) = v0_; A(C C) = C;",
    ///         &mut lex,
    ///         true,
    ///         &[]
    ///     ).expect("trs1");
    ///     let trs2 = trs1.smart_delete(1, 2);
    ///
    ///     assert_eq!(trs2.to_string(), "A(v0_ C) = v0_;");
    ///
    ///     let trs1 = parse_trs(
    ///         "A(v0_ v1_) = v0_; A(v0_ C) = v0_; A(C C) = C;",
    ///         &mut lex,
    ///         true,
    ///         &[]
    ///     ).expect("trs1");
    ///     let trs2 = trs1.smart_delete(0, 0);
    ///
    ///     assert_eq!(trs2.to_string(), "A(v0_ v1_) = v0_;");
    /// })
    /// ```
    pub fn smart_delete(mut self, start: usize, stop: usize) -> Self {
        let self_len = self.len();
        let mut rules = std::mem::replace(&mut self.utrs.rules, Vec::with_capacity(self_len));
        rules.reverse();
        for i in 0..rules.len() {
            let mut rule = rules.pop().expect("rule");
            let saved = match (start <= i, i < stop) {
                (false, true) => {
                    let n = self
                        .utrs
                        .rules
                        .iter()
                        .chain(rules.iter().rev().take(stop - i).skip(start - 1 - i))
                        .map(|r| r.lhs.variables().iter().map(|v| v.id()).max().unwrap_or(0))
                        .max()
                        .map(|n| n + 1)
                        .unwrap_or(0);
                    rule.offset(n);
                    rules
                        .iter()
                        .rev()
                        .take(stop - i)
                        .skip(start - 1 - i)
                        .all(|safe| Term::pmatch(&[(&rule.lhs, &safe.lhs)]).is_none())
                        && self
                            .utrs
                            .rules
                            .iter()
                            .all(|chosen| Term::pmatch(&[(&chosen.lhs, &rule.lhs)]).is_none())
                }
                (true, true) => true,
                (true, false) => {
                    let n = self
                        .utrs
                        .rules
                        .iter()
                        .map(|r| r.lhs.variables().iter().map(|v| v.id()).max().unwrap_or(0))
                        .max()
                        .map(|n| n + 1)
                        .unwrap_or(0);
                    rule.offset(n);
                    self.utrs
                        .rules
                        .iter()
                        .all(|chosen: &Rule| Term::pmatch(&[(&chosen.lhs, &rule.lhs)]).is_none())
                }
                _ => unreachable!(),
            };
            if saved {
                rule.canonicalize(&mut HashMap::new());
                self.utrs.rules.push(rule)
            }
        }
        self
    }
}

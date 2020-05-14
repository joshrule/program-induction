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
    pub fn smart_delete(
        &self,
        mut start: usize,
        mut stop: usize,
    ) -> Result<Self, SampleError<'ctx>> {
        let mut rules = &self.utrs.rules[..];
        if rules.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            let mut new_rules: Vec<Rule> = Vec::with_capacity(rules.len());
            while !rules.is_empty() {
                let mut new_rule = rules[0].clone();
                let n = new_rules
                    .iter()
                    .map(|r| r.lhs.variables().iter().map(|v| v.id()).max().unwrap_or(0))
                    .max()
                    .map(|n| n + 1)
                    .unwrap_or(0);
                new_rule.offset(n);
                if (start == 0 && stop > 0)
                    || (start > 0
                        && rules
                            .iter()
                            .skip(1)
                            .all(|rule| Term::pmatch(&[(&rule.lhs, &new_rule.lhs)]).is_none()))
                    || new_rules
                        .iter()
                        .all(|rule: &Rule| Term::pmatch(&[(&rule.lhs, &new_rule.lhs)]).is_none())
                {
                    // in safe zone or outside safe zone with useful rule
                    new_rule.canonicalize(&mut HashMap::new());
                    new_rules.push(new_rule);
                }
                start = 1.max(start) - 1;
                stop = 1.max(stop) - 1;
                rules = &rules[1..];
            }
            Ok(TRS::new_unchecked(
                &self.lex,
                self.is_deterministic(),
                self.background,
                new_rules,
            ))
        }
    }
}

use super::{super::as_result, SampleError, TRS};
use itertools::Itertools;
use rand::Rng;
use term_rewriting::{Rule, Term};

impl<'a> TRS<'a> {
    /// Delete a learned rule from the rewrite system.
    pub fn delete_rule(&self) -> Result<Vec<TRS<'a>>, SampleError> {
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
    ) -> Result<Vec<TRS<'a>>, SampleError> {
        let deletable = as_result(self.clauses())?;
        let mut trss = vec![];
        if 2usize.pow((1 + deletable.len()) as u32) - 2 > threshold {
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
    pub fn smart_delete(&self, mut start: usize, mut stop: usize) -> Result<TRS<'a>, SampleError> {
        let mut rules = &self.utrs.rules[..];
        if rules.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            let mut new_rules = Vec::with_capacity(rules.len());
            while !rules.is_empty() {
                let new_rule = rules[0].clone();
                if (start == 0 && stop > 0)
                    || (start > 0
                        && rules
                            .iter()
                            .skip(1)
                            .all(|rule| Term::pmatch(vec![(&rule.lhs, &new_rule.lhs)]).is_none()))
                    || new_rules
                        .iter()
                        .all(|rule: &Rule| Term::pmatch(vec![(&rule.lhs, &new_rule.lhs)]).is_none())
                {
                    // in safe zone or outside safe zone with useful rule
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

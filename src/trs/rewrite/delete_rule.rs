use super::{SampleError, TRS};
use itertools::Itertools;
use term_rewriting::{Rule, Term};

impl TRS {
    /// Delete a rule from the rewrite system, excluding background knowledge.
    pub fn delete_rule(&self) -> Result<Vec<TRS>, SampleError> {
        let background = &self.lex.0.read().expect("poisoned lexicon").background;
        let clauses = self.utrs.clauses();
        let deletable = clauses
            .iter()
            .filter(|c| !background.contains(c))
            .collect_vec();
        if deletable.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            deletable
                .iter()
                .map(|d| {
                    let mut trs = self.clone();
                    trs.utrs.remove_clauses(d)?;
                    Ok(trs)
                })
                .collect()
        }
    }

    /// Try deleting all combinations of rules from the rewrite system.
    pub fn delete_rules(self) -> Result<Vec<TRS>, SampleError> {
        let n_rules = self.num_learned_rules();
        let deletable = self.utrs.rules[0..n_rules]
            .iter()
            .flat_map(|r| r.clauses())
            .collect_vec();
        if deletable.is_empty() {
            Ok(vec![self])
        } else {
            let mut trss = vec![];
            for n in 1..=deletable.len() {
                for rules in deletable.iter().combinations(n) {
                    let mut trs = self.clone();
                    for rule in &rules {
                        trs.utrs.remove_clauses(rule)?;
                    }
                    trss.push(trs);
                }
            }
            trss.push(self);
            Ok(trss)
        }
    }

    /// Delete rules from the rewrite system which are outmatched by prior rules.
    pub fn smart_delete(&self, start: usize, stop: usize) -> Result<TRS, SampleError> {
        if self.num_learned_rules() == 0 {
            Err(SampleError::OptionsExhausted)
        } else {
            let mut trs = TRS::new_unchecked(&self.lex, vec![]);
            trs.utrs.rules.clear();
            let n = self.num_learned_rules();
            trs.smart_delete_helper(start, stop, &self.utrs.rules[0..n]);
            Ok(trs)
        }
    }

    fn smart_delete_helper(&mut self, start: usize, stop: usize, rules: &[Rule]) {
        if rules.is_empty() {
            let mut bg = self
                .lex
                .0
                .read()
                .expect("poisoned lexicon")
                .background
                .clone();
            self.utrs.rules.append(&mut bg);
        } else {
            let new_start = 1.max(start) - 1;
            let new_stop = 1.max(stop) - 1;
            let new_rule = rules[0].clone();
            if (start == 0 && stop > 0)
                || (start > 0
                    && rules
                        .iter()
                        .skip(1)
                        .all(|rule| Term::pmatch(vec![(&rule.lhs, &new_rule.lhs)]).is_none()))
                || self
                    .utrs
                    .rules
                    .iter()
                    .all(|rule| Term::pmatch(vec![(&rule.lhs, &new_rule.lhs)]).is_none())
            {
                // in safe zone or outside safe zone with useful rule
                self.utrs.rules.push(new_rule);
            }
            self.smart_delete_helper(new_start, new_stop, &rules[1..])
        }
    }
}

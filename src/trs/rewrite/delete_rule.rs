use super::{SampleError, TRS};
use itertools::Itertools;
use term_rewriting::{Rule, Term};

impl TRS {
    /// Delete a rule from the rewrite system if possible. Background knowledge
    /// cannot be deleted.
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

    pub fn smart_delete(&self, start: usize, stop: usize) -> Result<Vec<TRS>, SampleError> {
        if self.num_learned_rules() == 0 {
            Err(SampleError::OptionsExhausted)
        } else {
            let mut trs = TRS::new(&self.lex, vec![])?;
            trs.utrs.rules.truncate(0);
            let n = self.num_learned_rules();
            let trss = trs.smart_delete_helper(start, stop, &self.utrs.rules[0..n]);
            if trss.is_empty() {
                Err(SampleError::OptionsExhausted)
            } else {
                Ok(trss)
            }
        }
    }

    pub fn smart_delete_helper(&self, start: usize, stop: usize, rules: &[Rule]) -> Vec<TRS> {
        if rules.is_empty() {
            let mut trs = self.clone();
            let mut bg = self
                .lex
                .0
                .read()
                .expect("poisoned lexicon")
                .background
                .clone();
            trs.utrs.rules.append(&mut bg);
            vec![trs]
        } else {
            let new_start = 1.max(start) - 1;
            let new_stop = 1.max(stop) - 1;
            let new_rule = rules[0].clone();
            if start == 0 && stop > 0 {
                // in safe zone
                let mut trs = self.clone();
                trs.utrs.rules.push(new_rule);
                trs.smart_delete_helper(new_start, new_stop, &rules[1..])
            } else if self
                .utrs
                .rules
                .iter()
                .all(|rule| Term::pmatch(vec![(&rule.lhs, &new_rule.lhs)]).is_none())
            {
                // outside safe zone with useful rule
                let mut trs = self.clone();
                trs.utrs.rules.push(new_rule);
                let mut xs = self.smart_delete_helper(new_start, new_stop, &rules[1..]);
                let mut ys = trs.smart_delete_helper(new_start, new_stop, &rules[1..]);
                xs.append(&mut ys);
                xs
            } else {
                // outside safe zone with useless rule
                self.smart_delete_helper(new_start, new_stop, &rules[1..])
            }
        }
    }

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
            for n in 1..deletable.len() {
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

    pub fn delete_ruless(trss: Vec<TRS>) -> Result<Vec<TRS>, SampleError> {
        let mut results = vec![];
        for trs in trss {
            results.append(&mut trs.delete_rules()?);
        }
        Ok(results)
    }
}

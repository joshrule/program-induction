use itertools::Itertools;

use super::{SampleError, TRS};

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

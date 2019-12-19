use super::{Lexicon, SampleError, TRS};
use itertools::Itertools;
use term_rewriting::{Rule, Term};

impl TRS {
    /// Delete a learned rule from the rewrite system.
    pub fn delete_rule(&self) -> Result<Vec<TRS>, SampleError> {
        let n_rules = self.num_learned_rules();
        let deletable = self.utrs.rules[0..n_rules]
            .iter()
            .flat_map(|r| r.clauses())
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

    /// Delete all combinations of learned rules from the rewrite system.
    pub fn delete_rules(self) -> Result<Vec<TRS>, SampleError> {
        let n_rules = self.num_learned_rules();
        let deletable = self.utrs.rules[0..n_rules]
            .iter()
            .flat_map(|r| r.clauses())
            .collect_vec();
        if deletable.is_empty() {
            Err(SampleError::OptionsExhausted)
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

    /// Delete rules from the rewrite system whose LHS matches a prior rule's.
    pub fn smart_delete(&self, start: usize, stop: usize) -> Result<TRS, SampleError> {
        let rules = &self.utrs.rules[0..self.num_learned_rules()];
        if rules.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(TRS::smart_delete_helper(&self.lex, start, stop, rules))
        }
    }

    fn smart_delete_helper(
        lex: &Lexicon,
        mut start: usize,
        mut stop: usize,
        mut rules: &[Rule],
    ) -> TRS {
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
        TRS::new_unchecked(lex, new_rules)
    }
}

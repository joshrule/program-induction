use itertools::Itertools;
use rand::Rng;
use term_rewriting::{Context, Rule, RuleContext};

use super::{SampleError, TRS};

impl TRS {
    /// Regenerate some portion of a rule
    pub fn regenerate_rule<R: Rng>(
        &self,
        atom_weights: (f64, f64, f64, f64),
        max_size: usize,
        rng: &mut R,
    ) -> Result<Vec<TRS>, SampleError> {
        let (n, clause) = self.choose_clause(rng)?;
        let new_rules = self.regenerate_helper(&clause, atom_weights, max_size)?;
        let new_trss = new_rules
            .into_iter()
            .filter_map(|new_clause| {
                let mut trs = self.clone();
                let okay = trs.replace(n, &clause, new_clause).is_ok();
                if okay {
                    Some(trs)
                } else {
                    None
                }
            })
            .collect_vec();
        if new_trss.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(new_trss)
        }
    }
    fn regenerate_helper(
        &self,
        clause: &Rule,
        atom_weights: (f64, f64, f64, f64),
        max_size: usize,
    ) -> Result<Vec<Rule>, SampleError> {
        let rulecontext = RuleContext::from(clause.clone());
        let subcontexts = rulecontext.subcontexts();
        // sample one at random
        let mut rules = Vec::with_capacity(subcontexts.len());
        for subcontext in &subcontexts {
            // replace it with a hole
            let template = rulecontext.replace(&subcontext.1, Context::Hole).unwrap();
            // sample a term from the context
            let trs = self.clone();
            let new_result = trs
                .lex
                .sample_rule_from_context(template, atom_weights, true, max_size)
                .drop();
            if let Ok(new_clause) = new_result {
                if new_clause.lhs != new_clause.rhs().unwrap() {
                    rules.push(new_clause);
                }
            }
        }
        if rules.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(rules)
        }
    }
}

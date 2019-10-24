use itertools::Itertools;
use rand::Rng;
use std::collections::HashMap;
use term_rewriting::{Rule, Term};

use super::{SampleError, TRS};

impl TRS {
    /// Replace a subterm of the rule with a variable.
    pub fn replace_term_with_var<R: Rng>(&self, rng: &mut R) -> Result<Vec<TRS>, SampleError> {
        let old_trs = self.clone(); // TODO: cloning is a hack!
        let clause = self.choose_clause(rng)?;
        let mut types = HashMap::new();
        old_trs.lex.infer_rule(&clause, &mut types).drop()?;
        let new_trss = clause
            .lhs
            .subterms()
            .iter()
            .unique_by(|(t, _)| t)
            .filter_map(|(t, p)| {
                // need to indicate that p is from the LHS
                let mut real_p = p.clone();
                real_p.insert(0, 0);
                let mut trs = self.clone();
                let term_type = &types[&real_p];
                let new_var = trs.lex.invent_variable(term_type);
                let new_term = Term::Variable(new_var);
                let new_lhs = clause.lhs.replace_all(t, &new_term);
                let new_rhs = clause.rhs().unwrap().replace_all(t, &new_term);
                Rule::new(new_lhs, vec![new_rhs]).and_then(|new_clause| {
                    let okay = trs.replace(&clause, new_clause).is_ok();
                    if okay {
                        Some(trs)
                    } else {
                        None
                    }
                })
            })
            .collect_vec();
        if new_trss.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(new_trss)
        }
    }
}

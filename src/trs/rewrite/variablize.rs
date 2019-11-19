use itertools::Itertools;
use rand::Rng;
use std::collections::HashMap;
use term_rewriting::{Rule, Term, TRS as UntypedTRS};

use super::{SampleError, TRS};

impl TRS {
    /// Replace a subterm of the rule with a variable.
        let clause = self.choose_clause(rng)?;
    pub fn variablize<R: Rng>(&self, rng: &mut R) -> Result<Vec<TRS>, SampleError> {
        let mut types = HashMap::new();
        self.lex.infer_rule(&clause, &mut types).drop()?;
        let new_trss = clause
            .lhs
            .subterms()
            .iter()
            .unique_by(|(t, _)| t)
            .filter_map(|(t, p)| {
                let mut real_p = vec![0]; // indicates that p is from the LHS
                real_p.extend_from_slice(p);
                let mut trs = self.clone();
                let term_type = &types[&real_p];
                let new_var = trs.lex.invent_variable(term_type);
                let new_term = Term::Variable(new_var);
                let new_lhs = clause.lhs.replace_all(t, &new_term);
                let new_rhs = clause.rhs().unwrap().replace_all(t, &new_term);
                let new_clause = Rule::new(new_lhs, vec![new_rhs])?;
                trs.replace(&clause, new_clause).ok()?;
                if !UntypedTRS::alphas(&trs.utrs, &self.utrs) {
                    trs.smart_delete(n, n + 1).ok()
                } else {
                    None
                }
            })
            .flatten()
            .collect_vec();
        if new_trss.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(new_trss)
        }
    }
}

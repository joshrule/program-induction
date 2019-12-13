use super::{SampleError, TRS};
use itertools::Itertools;
use term_rewriting::{Rule, Term};

impl TRS {
    pub fn combine(parent1: &TRS, parent2: &TRS) -> Result<Vec<TRS>, SampleError> {
        TRS::merge(parent1, parent2)?
            .smart_delete(0, 0)?
            .delete_rules()
    }
    fn merge(trs1: &TRS, trs2: &TRS) -> Result<TRS, SampleError> {
        if trs1.lex != trs2.lex {
            return Err(SampleError::OptionsExhausted);
        }
        let mut rules = trs1.utrs.rules[..trs1.num_learned_rules()]
            .iter()
            .flat_map(Rule::clauses)
            .collect_vec();
        trs2.utrs.rules[..trs2.num_learned_rules()]
            .iter()
            .flat_map(Rule::clauses)
            .for_each(|r1| {
                let unique = !rules.iter().any(|r2| {
                    Rule::alpha(&r1, r2).is_some()
                        || (trs1.utrs.is_deterministic()
                            && Term::alpha(vec![(&r1.lhs, &r2.lhs)]).is_some())
                });
                if unique {
                    rules.push(r1);
                }
            });
        Ok(TRS::new(&trs1.lex, rules)?)
    }
}

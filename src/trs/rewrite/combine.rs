use super::{SampleError, TRS};
use itertools::Itertools;
use rand::Rng;
use term_rewriting::{Rule, Term};

impl<'a> TRS<'a> {
    pub fn combine<'b, R: Rng>(
        parent1: &TRS<'b>,
        parent2: &TRS,
        rng: &mut R,
        threshold: usize,
    ) -> Result<Vec<TRS<'b>>, SampleError> {
        TRS::merge(parent1, parent2)?
            .smart_delete(0, 0)?
            .delete_rules(rng, threshold)
    }
    fn merge<'b>(trs1: &TRS<'b>, trs2: &TRS) -> Result<TRS<'b>, SampleError> {
        if trs1.lex != trs2.lex {
            return Err(SampleError::OptionsExhausted);
        }
        let mut rules = trs1.utrs.rules.iter().flat_map(Rule::clauses).collect_vec();
        trs2.utrs
            .rules
            .iter()
            .flat_map(Rule::clauses)
            .for_each(|r1| {
                let unique = !rules.iter().any(|r2| {
                    Rule::alpha(&r1, r2).is_some()
                        || (trs1.is_deterministic()
                            && Term::alpha(vec![(&r1.lhs, &r2.lhs)]).is_some())
                });
                if unique {
                    rules.push(r1);
                }
            });
        Ok(TRS::new(
            &trs1.lex,
            trs1.is_deterministic(),
            trs1.background,
            rules,
        )?)
    }
}

use super::{SampleError, TRS};
use itertools::Itertools;
use rand::Rng;
use term_rewriting::{Rule, Term};

impl<'a, 'b> TRS<'a, 'b> {
    pub fn combine<'c, 'd, R: Rng>(
        parent1: &TRS<'c, 'd>,
        parent2: &TRS,
        rng: &mut R,
        threshold: usize,
    ) -> Result<Vec<TRS<'c, 'd>>, SampleError> {
        TRS::merge(parent1, parent2)?
            .smart_delete(0, 0)?
            .delete_rules(rng, threshold)
    }
    fn merge<'c, 'd>(trs1: &TRS<'c, 'd>, trs2: &TRS) -> Result<TRS<'c, 'd>, SampleError> {
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

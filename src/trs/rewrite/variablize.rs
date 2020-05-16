use itertools::Itertools;
use polytype::atype::Ty;
use std::{collections::HashMap, convert::TryFrom};
use term_rewriting::{Context, Place, Rule, RuleContext, Variable};
use trs::{Lexicon, TRS};

pub type Variablization<'ctx> = (usize, Ty<'ctx>, Vec<Place>);

fn find_variablizations<'ctx, 'b>(
    lex: &Lexicon<'ctx, 'b>,
    n: usize,
    rule: &Rule,
) -> Option<Vec<Variablization<'ctx>>> {
    // List the places where each term/type token occurs.
    let env = lex.infer_rule(rule).ok()?;
    let map = rule
        .subterms()
        .into_iter()
        .zip(env.tps.iter().map(|tp| tp.apply(&env.sub)))
        .filter(|((term, _), _)| term.as_application().is_some())
        .map(|((term, place), tp)| ((term, tp), place))
        .into_group_map()
        .into_iter()
        .filter(|(_, places)| places.iter().any(|place| place[0] == 0))
        .filter(|(_, places)| !places.contains(&vec![0]))
        .map(|((_, tp), places)| (n, tp, places))
        .collect_vec();
    Some(map)
}

impl<'ctx, 'b> TRS<'ctx, 'b> {
    pub fn find_all_variablizations(&self) -> Vec<Variablization<'ctx>> {
        let clauses = self.utrs.clauses();
        let self_len = clauses.len();
        clauses
            .iter()
            .enumerate()
            .filter_map(|(i, rule)| find_variablizations(&self.lex, i, rule))
            .flatten()
            .unique()
            .sorted_by_key(|(rule, _, places)| {
                let best_place = places.iter().filter(|place| place[0] == 0).max().unwrap();
                (self_len - *rule, best_place.clone())
            })
            .collect_vec()
    }
    pub fn apply_variablization(
        &self,
        tp: Ty<'ctx>,
        places: &[Place],
        rule: &Rule,
    ) -> Option<Rule> {
        places
            .get(0)
            .and_then(|place| rule.at(place))
            .and_then(|term| {
                let applies = {
                    let mut tp_rule = rule.clone();
                    tp_rule.canonicalize(&mut HashMap::new());
                    let env = self.lex.infer_rule(&tp_rule).ok()?;
                    tp_rule
                        .subterms()
                        .into_iter()
                        .zip(env.tps.iter().map(|tp| tp.apply(&env.sub)))
                        .all(|((_, p), t)| {
                            !places.contains(&p) || (t == tp) && rule.at(&p) == Some(term)
                        })
                };
                if applies {
                    let mut context =
                        RuleContext::from(rule.clone()).replace_all(places, Context::Hole)?;
                    context.canonicalize(&mut HashMap::new());
                    let id = context.lhs.variables().len();
                    let context = context.replace_all(places, Context::Variable(Variable(id)))?;
                    let mut new_rule = Rule::try_from(&context).ok()?;
                    new_rule.canonicalize(&mut HashMap::new());
                    Some(new_rule)
                } else {
                    None
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use polytype::atype::{with_ctx, TypeContext};
    use trs::parser::{parse_lexicon, parse_trs};
    use trs::Lexicon;

    fn create_test_lexicon<'b, 'ctx>(ctx: &TypeContext<'ctx>) -> Lexicon<'ctx, 'b> {
        parse_lexicon(
            &[
                "C/0: list -> list;",
                "CONS/0: nat -> list -> list;",
                "NIL/0: list;",
                "DECC/0: nat -> int -> nat;",
                "DIGIT/0: int -> nat;",
                "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
                "0/0: int; 1/0: int; 2/0: int;",
                "3/0: int; 4/0: int; 5/0: int;",
                "6/0: int; 7/0: int; 8/0: int;",
                "9/0: int;",
            ]
            .join(" "),
            &ctx,
        )
        .expect("parsed lexicon")
    }

    #[test]
    fn find_all_variablizations_test() {
        with_ctx(1024, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
            let trs = parse_trs(".(C .(.(CONS .(v0_ v1_)) NIL)) = NIL; .(C .(.(CONS .(v0_ v1_)) v2_)) = .(.(CONS .(v0_ v1_)) .(C v2_));", &mut lex, true, &[])
            .expect("trs");
            let vs = trs.find_all_variablizations();

            assert_eq!(
                vs.iter()
                    .map(|(n, tp, places)| format!("{} {} {:?}", n, tp, places))
                    .sorted()
                    .join(" "),
                [
                    "0 list [[0, 1, 1], [1]]",
                    "0 list [[0, 1]]",
                    "0 list → list [[0, 0]]",
                    "0 list → list [[0, 1, 0]]",
                    "0 nat [[0, 1, 0, 1]]",
                    "0 nat → list → list [[0, 1, 0, 0]]",
                    "1 list [[0, 1]]",
                    "1 list → list [[0, 0], [1, 1, 0]]",
                    "1 list → list [[0, 1, 0], [1, 0]]",
                    "1 nat [[0, 1, 0, 1], [1, 0, 1]]",
                    "1 nat → list → list [[0, 1, 0, 0], [1, 0, 0]]",
                ]
                .iter()
                .join(" ")
            );
        })
    }

    #[test]
    fn apply_variablization_test() {
        with_ctx(1024, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
            let trs = parse_trs(".(C .(.(CONS .(v0_ v1_)) NIL)) = NIL; .(C .(.(CONS .(v0_ v1_)) v2_)) = .(.(CONS .(v0_ v1_)) .(C v2_));", &mut lex, true, &[])
            .expect("parsed trs");
            let vs = trs.find_all_variablizations();
            let clauses = trs.utrs().clauses();
            let mut result_strings = vec![];
            for (n, tp, places) in vs
                .into_iter()
                .sorted_by_key(|(n, tp, places)| format!("{} {} {:?}", n, tp, places))
            {
                result_strings.push(format!("{} {} {:?}", n, tp, places));
                if let Some(new_r) = trs.apply_variablization(&tp, &places, &clauses[n]) {
                    result_strings.push(format!("{}", new_r.pretty(&trs.lex.signature())));
                }
            }
            assert_eq!(
                result_strings,
                [
                    "0 list [[0, 1, 1], [1]]",
                    "C (CONS (v0_ v1_) v2_) = v2_",
                    "0 list [[0, 1]]",
                    "C v0_ = []",
                    "0 list → list [[0, 0]]",
                    "v0_ (CONS (v1_ v2_) []) = []",
                    "0 list → list [[0, 1, 0]]",
                    "C (v0_ []) = []",
                    "0 nat [[0, 1, 0, 1]]",
                    "C (CONS v0_ []) = []",
                    "0 nat → list → list [[0, 1, 0, 0]]",
                    "C (v0_ (v1_ v2_) []) = []",
                    "1 list [[0, 1]]",
                    "1 list → list [[0, 0], [1, 1, 0]]",
                    "v0_ (CONS (v1_ v2_) v3_) = CONS (v1_ v2_) (v0_ v3_)",
                    "1 list → list [[0, 1, 0], [1, 0]]",
                    "C (v0_ v1_) = v0_ (C v1_)",
                    "1 nat [[0, 1, 0, 1], [1, 0, 1]]",
                    "C (CONS v0_ v1_) = CONS v0_ (C v1_)",
                    "1 nat → list → list [[0, 1, 0, 0], [1, 0, 0]]",
                    "C (v0_ (v1_ v2_) v3_) = v0_ (v1_ v2_) (C v3_)",
                ]
            );
        })
    }
}

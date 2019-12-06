use super::{Lexicon, SampleError, TRS};
use itertools::Itertools;
use polytype::Type;
use std::collections::HashMap;
use term_rewriting::{Place, Rule, Term};

impl TRS {
    /// Replace a subterm of the rule with a variable.
    pub fn variablize(&self, data: &[Rule]) -> Result<Vec<TRS>, SampleError> {
        let snapshot = self.lex.snapshot();
        // Find the unique variablizations across rules/data.
        let all_rules = self.clauses_for_learning(data)?;
        let variablizations = all_rules
            .iter()
            .filter_map(|rule| TRS::find_variablizations(rule, &self.lex))
            .flatten()
            .unique();
        // For each variablization
        let trss = variablizations
            .filter_map(|(term, tp, places)| {
                // For each rule/datum, try the variablization, and collect the successful.
                let rules = all_rules
                    .iter()
                    .filter_map(|rule| {
                        TRS::variablize_rule(term, &tp, &places, rule, &self.lex).map(|x| (x, rule))
                    })
                    .collect_vec();
                if !rules.is_empty() {
                    // Swap out the rules and smart-delete without a safe zone.
                    let mut trs = self.clone();
                    for (new_rule, old_rule) in &rules {
                        trs.swap(old_rule, new_rule.clone()).ok()?;
                    }
                    trs = trs.smart_delete(0, 0).ok()?;
                    // Create a hypothesis containing just the new rules.
                    let (just_rules, _): (Vec<Rule>, Vec<&Rule>) = rules.into_iter().unzip();
                    let trs2 = TRS::new_unchecked(&self.lex, just_rules)
                        .smart_delete(0, 0)
                        .ok()?;
                    // Return the unique results
                    if trs != trs2 {
                        Some(vec![trs, trs2])
                    } else {
                        Some(vec![trs])
                    }
                } else {
                    None
                }
            })
            .flatten()
            .collect_vec();
        // Return the vector of hypotheses.
        self.lex.rollback(snapshot);
        if trss.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(trss)
        }
    }
    pub fn find_variablizations<'a>(
        rule: &'a Rule,
        lex: &Lexicon,
    ) -> Option<Vec<(&'a Term, Type, Vec<Place>)>> {
        // Type the rule.
        let mut types = HashMap::new();
        lex.infer_rule(&rule, &mut types).drop().ok()?;
        // List the places where each term/type token occurs.
        // TODO: we're using type *equality* rather than unificiation, which
        //       potentially reduces the scope of variablization.
        let map = rule
            .subterms()
            .into_iter()
            .filter(|(term, _)| term.as_application().is_some())
            .map(|(term, place)| ((term, types.remove(&place).unwrap()), place))
            .into_group_map()
            .into_iter()
            .filter(|(_, places)| places.iter().any(|place| place[0] == 0))
            .map(|((term, tp), places)| (term, tp, places))
            .collect_vec();
        Some(map)
    }
    pub fn variablize_rule(
        term: &Term,
        tp: &Type,
        places: &[Place],
        rule: &Rule,
        lex: &Lexicon,
    ) -> Option<Rule> {
        let mut types = HashMap::new();
        lex.infer_rule(&rule, &mut types).drop().ok()?;
        if places
            .iter()
            .all(|place| types.get(place) == Some(tp) && rule.at(place) == Some(term))
        {
            rule.replace_all(places, Term::Variable(lex.invent_variable(tp)))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use polytype::Context as TypeContext;
    use trs::parser::{parse_lexicon, parse_rule, parse_trs};
    use trs::TRS;

    #[test]
    fn find_variablizations_test() {
        let lex = parse_lexicon(
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
            "",
            "",
            true,
            TypeContext::default(),
        )
        .expect("parsed lexicon");
        let rule = parse_rule(".(C .(.(CONS x_) NIL)) = .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) NIL))))))",
            &lex,
        )
            .expect("parsed trs");
        let opt = TRS::find_variablizations(&rule, &lex);
        assert!(opt.is_some());
        let variablizations = opt.unwrap();
        let vs = variablizations.iter().collect::<Vec<_>>();

        for (term, tp, places) in &vs {
            println!("{} {} {:?}\n\n", term.pretty(&lex.signature()), tp, places);
        }

        assert_eq!(vs.len(), 6);
    }

    #[test]
    fn variablize_test_1() {
        let lex = parse_lexicon(
            &[
                "+/2: INT -> INT -> INT;",
                " */2: INT -> INT -> INT;",
                " ^/2: INT -> INT -> INT;",
                " 0/0: INT; 1/0: INT; 2/0: INT;",
                " 3/0: INT; 4/0: INT; 6/0: INT;",
                " 9/0: INT;",
            ]
            .join(" "),
            "",
            "",
            true,
            TypeContext::default(),
        )
        .unwrap();
        let trs = parse_trs(
            "^(+(x_ 1) 2) = +(^(x_ 2) +(*(2 x_) 1)); ^(+(x_ 2) 2) = +(^(x_ 2) +(*(4 x_) 4)); ^(+(x_ 3) 2) = +(^(x_ 2) +(*(6 x_) 9));",
            &lex,
        )
            .expect("parsed trs");
        let trss = trs.variablize(&[]).unwrap();

        for trs in &trss {
            println!("{}\n", trs);
        }

        assert_eq!(trss.len(), 8);
    }

    #[test]
    fn variablize_test_2() {
        let lex = parse_lexicon(
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
            "",
            "",
            true,
            TypeContext::default(),
        )
        .unwrap();
        let trs = parse_trs(".(C .(.(CONS x_) NIL)) = .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) NIL))))));",
            &lex,
        )
            .expect("parsed trs");
        let trss = trs.variablize(&[]).unwrap();

        for trs in &trss {
            println!("{}\n", trs);
        }

        assert_eq!(trss.len(), 4);
    }

    #[test]
    fn variablize_test_3() {
        let lex = parse_lexicon(
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
            "",
            "",
            true,
            TypeContext::default(),
        )
        .expect("parsed lexicon");
        let rule = parse_rule(".(C .(.(CONS x_) NIL)) = .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) NIL))))))",
            &lex,
        )
            .expect("parsed rule");
        let trs = TRS::new_unchecked(&lex, vec![]);
        let trss = trs.variablize(&[rule]).unwrap();

        for trs in &trss {
            println!("{}\n", trs);
        }

        assert_eq!(trss.len(), 4);
    }

    #[test]
    fn variablize_test_4() {
        let lex = parse_lexicon(
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
            "",
            "",
            true,
            TypeContext::default(),
        )
        .expect("parsed lexicon");
        let rule = parse_rule(
            ".(C .(x_ NIL)) = .(x_ .(x_ .(x_ .(x_ .(x_ .(x_ NIL))))))",
            &lex,
        )
        .expect("parsed rule");
        let trs = TRS::new_unchecked(&lex, vec![]);
        let trss = trs.variablize(&[rule]).unwrap();

        for trs in &trss {
            println!("{}\n", trs);
        }

        assert_eq!(trss.len(), 2);
    }

    #[test]
    fn variablize_test_5() {
        let lex = parse_lexicon(
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
            "",
            "",
            true,
            TypeContext::default(),
        )
        .expect("parsed lexicon");
        let rule = parse_rule(
            "C (CONS (DIGIT 5) (CONS (DIGIT 3) (CONS (DIGIT 2) (CONS (DIGIT 9) (CONS (DIGIT 8) NIL))))) = CONS (DIGIT 2) NIL",
            &lex,
        )
        .expect("parsed rule");
        let trs = TRS::new_unchecked(&lex, vec![]);

        let gen1 = trs.variablize(&[rule]).unwrap();
        for trs in gen1.iter().sorted_by_key(|trs| trs.size()) {
            println!("{}\n", trs);
        }

        assert_eq!(gen1.len(), 24);

        let mut gen2 = gen1
            .iter()
            .sorted_by_key(|trs| trs.size())
            .take(50)
            .flat_map(|trs| trs.variablize(&[]).unwrap())
            .collect_vec();
        let mut i = 1;
        while i < gen2.len() {
            if !(0..i).any(|n| TRS::is_alpha(&gen2[n], &gen2[i])) {
                i += 1;
            } else {
                gen2.swap_remove(i);
            }
        }
        println!(">>> {}\n", gen2.len());
        for trs in gen2.iter().sorted_by_key(|trs| trs.size()).take(50) {
            println!("{}\n", trs);
        }
        assert_eq!(gen2.len(), 214);

        let mut gen3 = gen2
            .iter()
            .sorted_by_key(|trs| trs.size())
            .take(50)
            .filter_map(|trs| trs.variablize(&[]).ok())
            .flatten()
            .collect_vec();
        let mut i = 1;
        while i < gen3.len() {
            if !(0..i).any(|n| TRS::is_alpha(&gen3[n], &gen3[i])) {
                i += 1;
            } else {
                gen3.swap_remove(i);
            }
        }
        println!(">>> {}\n", gen3.len());
        for trs in gen3.iter().sorted_by_key(|trs| trs.size()).take(50) {
            println!("{}\n", trs);
        }

        assert_eq!(gen3.len(), 292);

        let mut gen4 = gen3
            .iter()
            .sorted_by_key(|trs| trs.size())
            .take(50)
            .filter_map(|trs| trs.variablize(&[]).ok())
            .flatten()
            .collect_vec();
        let mut i = 1;
        while i < gen4.len() {
            if !(0..i).any(|n| TRS::is_alpha(&gen4[n], &gen4[i])) {
                i += 1;
            } else {
                gen4.swap_remove(i);
            }
        }
        println!(">>> {}\n", gen4.len());
        for trs in gen4.iter().sorted_by_key(|trs| trs.size()).take(50) {
            println!("{}\n", trs);
        }

        assert_eq!(gen4.len(), 105);

        let mut gen5 = gen4
            .iter()
            .sorted_by_key(|trs| trs.size())
            .take(50)
            .filter_map(|trs| trs.variablize(&[]).ok())
            .flatten()
            .collect_vec();
        let mut i = 1;
        while i < gen5.len() {
            if !(0..i).any(|n| TRS::is_alpha(&gen5[n], &gen5[i])) {
                i += 1;
            } else {
                gen5.swap_remove(i);
            }
        }
        println!(">>> {}\n", gen5.len());
        for trs in gen5.iter().sorted_by_key(|trs| trs.size()).take(50) {
            println!("{}\n", trs);
        }

        assert_eq!(gen5.len(), 2004);
    }
}

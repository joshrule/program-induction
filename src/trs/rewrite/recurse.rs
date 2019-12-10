use super::{Lexicon, SampleError, TRS};
use itertools::Itertools;
use polytype::Type;
use std::collections::HashMap;
use term_rewriting::{Operator, Place, Rule, Term};
use utils::weighted_permutation;

impl TRS {
    pub fn recurse(&self, data: &[Rule], n_sampled: usize) -> Result<Vec<TRS>, SampleError> {
        let all_rules = self.clauses_for_learning(data)?;
        let snapshot = self.lex.snapshot();
        let op = self
            .lex
            .has_op(Some("."), 2)
            .map_err(|_| SampleError::OptionsExhausted)?;
        let (trss, ns): (Vec<_>, Vec<_>) = all_rules
            .iter()
            .flat_map(|r| TRS::find_transforms(r, &self.lex))
            .unique()
            .filter_map(|(f, lhs_place, rhs_place, tp)| {
                let mut trs = self.clone();
                let new_ruless: Vec<_> = all_rules
                    .windows(1)
                    .filter_map(|rules| {
                        let sub_new_rules = TRS::transform(
                            &f, &lhs_place, &rhs_place, op, &rules[0], &self.lex, &tp,
                        )
                        .unwrap_or_else(|_| vec![]);
                        if sub_new_rules.len() <= 1 {
                            None
                        } else {
                            trs.remove_clauses(rules).unwrap();
                            Some(sub_new_rules)
                        }
                    })
                    .collect();
                let mut new_bases = vec![];
                let mut new_recursives = vec![];
                for mut new_rules in new_ruless {
                    // put base case in  new bases
                    let new_base = new_rules.swap_remove(0);
                    if new_bases
                        .iter()
                        .all(|c| Rule::alpha(&c, &new_base).is_none())
                    {
                        new_bases.push(new_base);
                    }
                    // put each recursive in new recursive
                    for new_recursive in new_rules {
                        if new_recursives
                            .iter()
                            .all(|c| Rule::alpha(&c, &new_recursive).is_none())
                        {
                            new_recursives.push(new_recursive);
                        }
                    }
                }
                let mut new_rules = vec![];
                new_rules.append(&mut new_bases);
                new_rules.append(&mut new_recursives);
                let n = new_rules.len();
                if n == 1 {
                    return None;
                }
                trs.prepend_clauses(new_rules.clone()).ok()?;
                let trs = trs.smart_delete(0, n).ok()?;
                let trs2 = TRS::new(&self.lex, new_rules).unwrap();
                Some(vec![
                    (trs, (1.5 as f64).powi(n as i32)),
                    (trs2, (1.5 as f64).powi(n as i32)),
                ])
            })
            .flatten()
            .unzip();
        self.lex.rollback(snapshot);
        // HACK: 20 is a constant but not a magic number.
        let new_trss = weighted_permutation(&trss, &ns, Some(n_sampled));
        Ok(new_trss)
    }
    fn collect_recursive_fns<'a>(
        map: &HashMap<Place, Type>,
        lex: &Lexicon,
        rule: &'a Rule,
    ) -> Vec<(&'a Term, Place, Type)> {
        let mut headmost = vec![0];
        while map.contains_key(&headmost) {
            headmost.push(0);
        }
        headmost.pop();
        let mut fns = vec![];
        for i_place in 1..=headmost.len() {
            let k = &headmost[0..i_place];
            let v = &map[k];
            if let Some(tp) = v.returns() {
                let tps = v.args().unwrap();
                let snapshot = lex.snapshot();
                if tps.len() == 1 && lex.unify(tp, tps[0]).is_ok() {
                    let new_tp = tp.apply(&lex.0.read().expect("poisoned lexicon").ctx);
                    fns.push((rule.at(k).unwrap(), k.to_vec(), new_tp));
                }
                lex.rollback(snapshot);
            }
        }
        fns
    }
    fn partition_subrules(rule: &Rule) -> (Vec<Place>, Vec<Place>) {
        rule.subterms()
            .into_iter()
            .skip(1)
            .map(|x| x.1)
            .partition(|x| x[0] == 0)
    }
    /// This function returns a `Vec` of (f, lhs_place, rhs_place)
    /// - f: some potentially recursive function (i.e. f: a -> a)
    /// - lhs_place: some LHS subterm which could be f's input (lhs_place: a)
    /// - rhs_place: some RHS subterm which could be f's output (rhs_place: a)
    fn find_transforms(rule: &Rule, lex: &Lexicon) -> Vec<(Term, Place, Place, Type)> {
        let mut map = HashMap::new();
        if lex.infer_rule(rule, &mut map).keep().is_err() {
            return vec![];
        }
        let fs = TRS::collect_recursive_fns(&map, lex, rule);
        let (lhss, rhss) = TRS::partition_subrules(rule);
        let mut transforms = vec![];
        for (f, place, tp) in fs {
            for lhs_place in &lhss {
                let outer_snapshot = lex.snapshot();
                let diff_place = place != *lhs_place;
                let trivial = lhs_place.starts_with(&place[0..place.len() - 1])
                    && lhs_place.ends_with(&[1])
                    && lhs_place.len() == place.len();
                if diff_place && !trivial && lex.unify(&tp, &map[lhs_place]).is_ok() {
                    for rhs_place in &rhss {
                        let inner_snapshot = lex.snapshot();
                        if lex.unify(&tp, &map[rhs_place]).is_ok() {
                            let tp = tp.apply(&lex.0.read().expect("poisoned lexicon").ctx);
                            transforms.push((f.clone(), lhs_place.clone(), rhs_place.clone(), tp));
                        }
                        lex.rollback(inner_snapshot);
                    }
                }
                lex.rollback(outer_snapshot);
            }
        }
        transforms
    }
    fn transform(
        f: &Term,
        lhs_place: &[usize],
        rhs_place: &[usize],
        op: Operator,
        rule: &Rule,
        lex: &Lexicon,
        tp: &Type,
    ) -> Result<Vec<Rule>, SampleError> {
        let mut new_rules = vec![rule.clone()];
        // Iterate until failure.
        while let Ok((new_rule1, new_rule2)) = TRS::transform_inner(
            f,
            lhs_place,
            rhs_place,
            op,
            &new_rules[new_rules.len() - 1],
            lex,
            tp,
        ) {
            // Exit early if the LHS == the RHS; or the base case hasn't changed.
            if new_rule1.lhs == new_rule1.rhs().unwrap() {
                break;
            } else if let Some(old_rule) = new_rules.pop() {
                if Rule::alpha(&old_rule, &new_rule2).is_some() {
                    break;
                }
            }
            // Only add new recursive cases if they're alpha-unique.
            let unique = new_rules
                .iter()
                .all(|r| Rule::alpha(r, &new_rule1).is_none());
            if unique {
                new_rules.push(new_rule1);
            }
            new_rules.push(new_rule2);
            // HACK: 12 is a constant, but not a magic number
            if new_rules.len() > 12 {
                break;
            }
        }
        // Return
        if new_rules.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            new_rules.reverse();
            Ok(new_rules)
        }
    }
    fn transform_inner(
        f: &Term,
        lhs_place: &[usize],
        rhs_place: &[usize],
        op: Operator,
        rule: &Rule,
        lex: &Lexicon,
        var_type: &Type,
    ) -> Result<(Rule, Rule), SampleError> {
        // Perform necessary checks:
        // 0. the rule typechecks;
        let snapshot = lex.snapshot();
        let mut map = HashMap::new();
        if let Err(e) = lex.infer_rule(rule, &mut map).keep() {
            lex.rollback(snapshot);
            return Err(SampleError::from(e));
        }
        // 1. f is at the head of the LHS;
        TRS::leftmost_symbol_matches(&rule, f, lex, &map)?;
        // 2. lhs_place exists and has the appropriate type; and
        let lhs_structure = TRS::check_place(rule, lhs_place, var_type, lex, &map)?;
        // 3. rhs_place exists and has the appropriate type.
        let rhs_structure = TRS::check_place(rule, rhs_place, var_type, lex, &map)?;
        let new_var = lex.invent_variable(var_type);
        // Swap lhs_structure for: new_var.
        let mut new_rule1 = rule
            .replace(lhs_place, Term::Variable(new_var))
            .ok_or(SampleError::Subterm)?;
        // Swap rhs_structure for: f new_var.
        let new_subterm = Term::Application {
            op,
            args: vec![f.clone(), Term::Variable(new_var)],
        };
        new_rule1 = new_rule1
            .replace(rhs_place, new_subterm)
            .ok_or(SampleError::Subterm)?;
        // Create the rule: f lhs_structure = rhs_structure.
        let new_lhs = Term::Application {
            op,
            args: vec![f.clone(), lhs_structure.clone()],
        };
        let new_rule2 =
            Rule::new(new_lhs, vec![rhs_structure.clone()]).ok_or(SampleError::Subterm)?;
        // Return.
        lex.rollback(snapshot);
        Ok((new_rule1, new_rule2))
    }
    fn leftmost_symbol_matches(
        rule: &Rule,
        f: &Term,
        lex: &Lexicon,
        map: &HashMap<Place, Type>,
    ) -> Result<(), SampleError> {
        TRS::collect_recursive_fns(map, lex, rule)
            .iter()
            .find(|(term, _, _)| Term::alpha(vec![(term, f)]).is_some())
            .map(|_| ())
            .ok_or(SampleError::Subterm)
    }
    fn check_place<'a>(
        rule: &'a Rule,
        place: &[usize],
        tp: &Type,
        lex: &Lexicon,
        map: &HashMap<Place, Type>,
    ) -> Result<&'a Term, SampleError> {
        map.get(place)
            .and_then(|place_tp| lex.unify(tp, place_tp).ok())
            .and_then(|_| rule.at(place))
            .ok_or(SampleError::Subterm)
    }
}

#[cfg(test)]
mod tests {
    use super::TRS;
    use polytype::Context as TypeContext;
    use std::collections::HashMap;
    use trs::parser::{parse_lexicon, parse_rule, parse_term, parse_trs};

    #[test]
    fn find_transforms_test() {
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
        let rule = parse_rule(
            "C (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2 ) (CONS (DIGIT 0) NIL)))) = (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2 ) NIL)))",
            &lex,
        )
            .expect("parsed rule");
        let transforms = TRS::find_transforms(&rule, &lex);
        for (t, p1, p2, tp) in &transforms {
            println!("{} {:?} {:?} {}", t.pretty(&lex.signature()), p1, p2, tp);
        }

        assert_eq!(16, transforms.len());
    }

    #[test]
    fn find_transforms_test_2() {
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
        let rule = parse_rule(
            "C (CONS (DIGIT 6) (CONS (DECC (DIGIT 6) 3) (CONS (DECC (DIGIT 8) 6 ) (CONS (DIGIT 8) (CONS (DIGIT 9) (CONS (DIGIT 4) (CONS (DIGIT 7) NIL))))))) = (CONS (DIGIT 6) (CONS (DIGIT 6) (CONS (DIGIT 6) (CONS (DIGIT 6) (CONS (DECC (DIGIT 6) 3) (CONS (DECC (DIGIT 8) 6 ) (CONS (DIGIT 8) (CONS (DIGIT 9) (CONS (DIGIT 4) (CONS (DIGIT 7) NIL))))))))))",
            &lex,
        )
            .expect("parsed rule");
        let transforms = TRS::find_transforms(&rule, &lex);
        for (t, p1, p2, tp) in &transforms {
            println!("{} {:?} {:?} {}", t.pretty(&lex.signature()), p1, p2, tp);
        }

        assert_eq!(77, transforms.len());
    }

    #[test]
    fn transform_inner_test() {
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
        let op = lex.has_op(Some("."), 2).unwrap();
        let rule = parse_rule(
            "C (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2 ) (CONS (DIGIT 0) NIL)))) = (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2 ) NIL)))",
            &lex,
        )
            .expect("parsed rule");
        let f = parse_term("C", &lex).expect("parsed term");
        let lhs_place = vec![0, 1, 1];
        let rhs_place = vec![1, 1];
        let mut map = HashMap::new();
        lex.infer_rule(&rule, &mut map);
        let result = TRS::transform_inner(
            &f,
            &lhs_place,
            &rhs_place,
            op,
            &rule,
            &lex,
            &map[&lhs_place],
        );
        assert!(result.is_ok());
        let (new_rule1, new_rule2) = result.unwrap();
        let sig = &lex.0.read().expect("poisoned lexicon").signature;

        assert_eq!(
            "C (CONS (DIGIT 9) var0_) = CONS (DIGIT 9) (C var0_)",
            new_rule1.pretty(sig),
        );
        assert_eq!(
            "C (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2) (CONS (DIGIT 0) []))) = CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2) [])",
            new_rule2.pretty(sig),
        );
    }

    #[test]
    fn transform_test() {
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
        let op = lex.has_op(Some("."), 2).unwrap();
        let rule = parse_rule(
            "C (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) (CONS (DECC (DIGIT 5) 4) NIL))))))))) = (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) NIL))))))))",
            &lex,
        )
            .expect("parsed rule");
        let f = parse_term("C", &lex).expect("parsed term");
        let lhs_place = vec![0, 1, 1];
        let rhs_place = vec![1, 1];
        let tp = tp![list];
        let result = TRS::transform(&f, &lhs_place, &rhs_place, op, &rule, &lex, &tp);
        assert!(result.is_ok());
        let new_rules = result.unwrap();
        let sig = &lex.0.read().expect("poisoned lexicon").signature;

        assert_eq!(8, new_rules.len());
        assert_eq!(
            "C (CONS (DECC (DIGIT 5) 4) []) = []",
            new_rules[0].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DIGIT 0) var7_) = CONS (DIGIT 0) (C var7_)",
            new_rules[1].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DECC (DIGIT 7) 7) var6_) = CONS (DECC (DIGIT 7) 7) (C var6_)",
            new_rules[2].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DECC (DIGIT 2) 0) var4_) = CONS (DECC (DIGIT 2) 0) (C var4_)",
            new_rules[3].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DIGIT 9) var3_) = CONS (DIGIT 9) (C var3_)",
            new_rules[4].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DECC (DIGIT 1) 0) var2_) = CONS (DECC (DIGIT 1) 0) (C var2_)",
            new_rules[5].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DIGIT 3) var1_) = CONS (DIGIT 3) (C var1_)",
            new_rules[6].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DIGIT 2) var0_) = CONS (DIGIT 2) (C var0_)",
            new_rules[7].pretty(sig),
        );
    }
    #[test]
    fn recurse_1_test() {
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
        let trs = parse_trs(
            "C (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) (CONS (DECC (DIGIT 5) 4) NIL))))))))) = (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) NIL))))))));",
            &lex,
        )
            .expect("parsed TRS");
        let result = trs.recurse(&[], 20);
        assert!(result.is_ok());
        let trss = result.unwrap();

        for trs in &trss {
            println!("##\n{}\n##", trs);
        }

        assert_eq!(20, trss.len());
    }
    #[test]
    fn recurse_2_test() {
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
        let trs = parse_trs(
            "C (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) (CONS (DECC (DIGIT 5) 4) NIL))))))))) = (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) NIL))))))));C (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2) (CONS (DIGIT 0) NIL)))) = (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2 ) NIL)));",
            &lex,
        )
            .expect("parsed TRS");
        let result = trs.recurse(&[], 20);
        assert!(result.is_ok());
        let trss = result.unwrap();

        for trs in &trss {
            println!("##\n{}\n##", trs);
        }

        assert_eq!(20, trss.len());
    }
}

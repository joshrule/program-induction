use super::{Lexicon, SampleError, TRS};
use polytype::Type;
use std::collections::HashMap;
use term_rewriting::{Operator, Place, Rule, Term};
use utils::weighted_permutation;

impl TRS {
    pub fn recurse(&self, data: &[Rule]) -> Result<Vec<TRS>, SampleError> {
        let all_rules = self.clauses_for_learning(data)?;
        let snapshot = self.lex.snapshot();
        let op = self
            .lex
            .has_op(Some("."), 2)
            .map_err(|_| SampleError::OptionsExhausted)?;
        let (trss, ns): (Vec<_>, Vec<_>) = all_rules
            .iter()
            .flat_map(|r| TRS::find_transforms(r, &self.lex))
            .filter_map(|(f, lhs_place, rhs_place, rule)| {
                let new_rules =
                    TRS::transform(&f, &lhs_place, &rhs_place, op, rule, &self.lex).ok()?;
                let n = new_rules.len();
                if n == 1 {
                    return None;
                }
                let mut trs = self.clone();
                trs.remove_clauses(&[rule.clone()]).ok()?;
                trs.prepend_clauses(new_rules.clone()).ok()?;
                let trs = trs.smart_delete(0, new_rules.len()).ok()?;
                Some((trs, (1.5 as f64).powi(n as i32)))
            })
            .unzip();
        self.lex.rollback(snapshot);
        // HACK: 20 is a constant but not a magic number.
        Ok(weighted_permutation(&trss, &ns, Some(20)))
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
    /// This function returns a `Vec` of (f, lhs_place, rhs_place, rule)
    /// - f: some potentially recursive function (i.e. f: a -> a)
    /// - lhs_place: some LHS subterm which could be f's input (lhs_place: a)
    /// - rhs_place: some RHS subterm which could be f's output (rhs_place: a)
    /// - rule: a rule containing all these elements
    fn find_transforms<'a>(rule: &'a Rule, lex: &Lexicon) -> Vec<(Term, Place, Place, &'a Rule)> {
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
                            transforms.push((
                                f.clone(),
                                lhs_place.clone(),
                                rhs_place.clone(),
                                rule,
                            ));
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
    ) -> Result<Vec<Rule>, SampleError> {
        let mut map = HashMap::new();
        lex.infer_rule(rule, &mut map).keep()?;
        let mut new_rules = vec![rule.clone()];
        // Iterate until failure.
        while let Ok((new_rule1, new_rule2)) = TRS::transform_inner(
            f,
            lhs_place,
            rhs_place,
            op,
            &new_rules[new_rules.len() - 1],
            lex,
            &map,
        ) {
            // exit early if the LHS == the RHS
            if new_rule1.lhs == new_rule1.rhs().unwrap() {
                break;
            }
            if let Some(old_rule) = new_rules.pop() {
                if Rule::alpha(&old_rule, &new_rule2).is_some() {
                    break;
                }
            }
            // only add new recursive cases if they're alpha-unique
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
        map: &HashMap<Place, Type>,
    ) -> Result<(Rule, Rule), SampleError> {
        let lhs_structure = rule.at(lhs_place).ok_or(SampleError::Subterm)?.clone();
        let rhs_structure = rule.at(rhs_place).ok_or(SampleError::Subterm)?.clone();
        let new_var = lex.invent_variable(&map[lhs_place]);
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
            args: vec![f.clone(), lhs_structure],
        };
        let new_rule2 = Rule::new(new_lhs, vec![rhs_structure]).ok_or(SampleError::Subterm)?;
        // Return.
        Ok((new_rule1, new_rule2))
    }
}

#[cfg(test)]
mod tests {
    use super::TRS;
    use polytype::Context as TypeContext;
    use std::collections::HashMap;
    use trs::parser::{parse_lexicon, parse_rule, parse_term};

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
        for (t, p1, p2, _) in &transforms {
            println!("{} {:?} {:?}", t.pretty(&lex.signature()), p1, p2);
        }

        assert_eq!(16, transforms.len());
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
        let result = TRS::transform_inner(&f, &lhs_place, &rhs_place, op, &rule, &lex, &map);
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
        let mut map = HashMap::new();
        lex.infer_rule(&rule, &mut map);
        let result = TRS::transform(&f, &lhs_place, &rhs_place, op, &rule, &lex);
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
}

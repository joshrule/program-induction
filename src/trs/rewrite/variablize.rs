use super::{super::as_result, Lexicon, SampleError, TRS};
use itertools::Itertools;
use polytype::Type;
use std::collections::HashMap;
use term_rewriting::{Place, Rule, Term};

type Variablization = (Type, Vec<Place>);
type Transformation = (Rule, Rule);
type Rules = Vec<Rule>;
type Cluster<'a> = (Rules, Vec<&'a Variablization>);

impl<'a, 'b> TRS<'a, 'b> {
    /// Replace subterms of [`term_rewriting::Rule`]s with [`term_rewriting::Variable`]s.
    ///
    /// [`term_rewriting::Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
    /// [`term_rewriting::Variable`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Variable.html

    pub fn variablize(&self) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        // Generate the list of variablizations.
        let mut trs = self.clone();
        let mut trss = vec![];
        // Record which rules are affected by which variablizations.
        trs.analyze_variablizations(&trs.find_all_variablizations())
            // Group the results by the affected rules.
            .into_iter()
            .map(|(v, rs)| (rs, v))
            .into_group_map()
            // Augment as appropriate.
            .into_iter()
            .for_each(|cluster| trs.augment_trss(&mut trss, cluster));
        // Return the solution set.
        as_result(trss)
    }
    fn augment_trss(&self, trss: &mut Vec<TRS<'a, 'b>>, (old_rules, mut vars): Cluster) {
        // Sort variablizations by the lexicographically deepest place they affect.
        vars.sort_by_key(|(_, places)| places.iter().max().unwrap());
        vars.reverse();
        // For each variablization, v:
        for (i, (tp, places)) in vars.iter().enumerate() {
            // Try the root variablization, v:
            let mut trs = self.clone();
            if let Some(mut new_rules) = trs.apply_variablization(tp, places, &old_rules) {
                if let Some(optimized) = trs
                    .clone()
                    .adopt_solution(&mut TRS::make_solution(&old_rules, &new_rules))
                {
                    if optimized.unique_shape(trss) {
                        trss.push(optimized);
                        // Apply each remaining variablization, v', if possible.
                        for (tp, places) in vars.iter().skip(i + 1) {
                            if let Some(newer_rules) =
                                trs.apply_variablization(tp, places, &new_rules)
                            {
                                new_rules = newer_rules;
                            }
                        }
                        // Adopt the result.
                        if let Some(optimized) =
                            trs.adopt_solution(&mut TRS::make_solution(&old_rules, &new_rules))
                        {
                            if optimized.unique_shape(trss) {
                                trss.push(optimized);
                            }
                        }
                    }
                }
            }
        }
    }
    fn find_all_variablizations(&self) -> Vec<(Type, Vec<Place>)> {
        self.clauses()
            .iter()
            .filter_map(|(_, rule)| TRS::find_variablizations(rule, &self.lex))
            .flatten()
            .unique()
            .collect_vec()
    }
    fn find_variablizations(rule: &Rule, lex: &Lexicon) -> Option<Vec<(Type, Vec<Place>)>> {
        // Type the rule.
        let mut types = HashMap::new();
        lex.infer_rule(&rule, &mut types).drop().ok()?;
        // List the places where each term/type token occurs.
        let map = rule
            .subterms()
            .into_iter()
            .filter(|(term, _)| term.as_application().is_some())
            .map(|(term, place)| ((term, types.remove(&place).unwrap()), place))
            .into_group_map()
            .into_iter()
            .filter(|(_, places)| places.iter().any(|place| place[0] == 0))
            .filter(|(_, places)| !places.contains(&vec![0]))
            .map(|((_, tp), places)| (tp, places))
            .collect_vec();
        Some(map)
    }
    fn apply_variablization(
        &mut self,
        tp: &Type,
        places: &[Place],
        rules: &[Rule],
    ) -> Option<Vec<Rule>> {
        let mut new_rules = Vec::with_capacity(rules.len());
        for rule in rules {
            match self.apply_variablization_to_rule(tp, places, rule) {
                Some(new_rule) => new_rules.push(new_rule),
                None => return None,
            }
        }
        Some(new_rules)
    }
    fn apply_variablization_to_rule(
        &mut self,
        tp: &Type,
        places: &[Place],
        rule: &Rule,
    ) -> Option<Rule> {
        let mut types = HashMap::new();
        self.lex.infer_rule(&rule, &mut types).drop().ok()?;
        places
            .get(0)
            .and_then(|place| rule.at(place))
            .and_then(|term| {
                if places
                    .iter()
                    .all(|place| types.get(place) == Some(tp) && rule.at(place) == Some(term))
                {
                    rule.replace_all(places, Term::Variable(self.lex.invent_variable(tp)))
                } else {
                    None
                }
            })
    }
    fn analyze_variablizations<'c>(
        &mut self,
        vs: &'c [(Type, Vec<Place>)],
    ) -> Vec<(&'c Variablization, Vec<Rule>)> {
        vs.iter()
            .map(|v| (v, self.analyze_variablization(&v.0, &v.1)))
            .filter(|(_, rs)| !rs.is_empty())
            .collect_vec()
    }
    fn analyze_variablization(&mut self, tp: &Type, places: &[Place]) -> Vec<Rule> {
        self.clauses()
            .into_iter()
            .map(|(_, rule)| rule)
            .filter(|rule| self.analyze_variablization_for_rule(tp, places, &rule))
            .collect_vec()
    }
    fn analyze_variablization_for_rule(
        &mut self,
        tp: &Type,
        places: &[Place],
        rule: &Rule,
    ) -> bool {
        let mut types = HashMap::new();
        self.lex.infer_rule(&rule, &mut types).drop().is_ok()
            && places
                .get(0)
                .and_then(|place| rule.at(place))
                .map(|term| {
                    let mut rule_lhs_sts = rule
                        .lhs
                        .subterms()
                        .into_iter()
                        .map(|(x, _)| x)
                        .collect_vec();
                    let term_sts = rule
                        .lhs
                        .subterms()
                        .into_iter()
                        .map(|(x, _)| x)
                        .collect_vec();
                    let mut i = 0;
                    while i < rule_lhs_sts.len() {
                        if term_sts.contains(&rule_lhs_sts[i]) {
                            rule_lhs_sts.swap_remove(0);
                        } else {
                            i += 1;
                        }
                    }
                    let duplicate_case = term
                        .variables()
                        .iter()
                        .all(|v| rule_lhs_sts.contains(&&Term::Variable(*v)));
                    let unused_case = rule.rhs.iter().all(|rhs| {
                        term.variables()
                            .iter()
                            .all(|v| !rhs.variables().contains(v))
                    });
                    (unused_case || duplicate_case)
                        && places.iter().all(|place| {
                            types.get(place) == Some(tp) && rule.at(place) == Some(term)
                        })
                })
                .unwrap_or(false)
        // can_remove: places doesn't include rhs or variables in term occur elsewhere in LHS
    }
    fn make_solution(old: &[Rule], new: &[Rule]) -> Vec<Transformation> {
        new.iter().cloned().zip(old.iter().cloned()).collect_vec()
    }
    fn adopt_solution(mut self, rules: &mut Vec<Transformation>) -> Option<TRS<'a, 'b>> {
        let mut i = 0;
        // Uniquify by alpha-equivalent rules.
        while i < rules.len() {
            if rules
                .iter()
                .take(i)
                .any(|(r, _)| Rule::alpha(r, &rules[i].0).is_some())
            {
                rules.remove(i);
            } else {
                i += 1;
            }
        }
        // Remove rules sharing an LHS if deterministic.
        if self.is_deterministic() {
            i = 0;
            while i < rules.len() {
                let mut okay = true;
                let mut j = i + 1;
                while j < rules.len() {
                    if Term::alpha(vec![(&rules[j].0.lhs, &rules[i].0.lhs)]).is_some() {
                        okay = false;
                        rules.remove(j);
                    } else {
                        j += 1;
                    }
                }
                if !okay {
                    rules.remove(i);
                } else {
                    i += 1;
                }
            }
        }
        // Create a new TRS.
        if rules.is_empty() {
            None
        } else {
            self.variablize_filter_background(rules);
            self.swap_rules(rules).ok()?;
            self.smart_delete(0, 0).ok()
        }
    }
    fn variablize_filter_background(&self, xs: &mut Vec<Transformation>) {
        for (n, _) in xs.iter_mut() {
            for bg in self.background {
                n.discard(bg);
            }
        }
        if self.utrs.is_deterministic() {
            xs.retain(|(n, _)| {
                self.background
                    .iter()
                    .all(|bg| Term::alpha(vec![(&bg.lhs, &n.lhs)]).is_none())
            });
        }
        xs.retain(|(n, _)| !n.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use polytype::Context as TypeContext;
    use trs::parser::{parse_lexicon, parse_rule, parse_trs};
    use trs::{Lexicon, TRS};

    fn create_test_lexicon<'b>() -> Lexicon<'b> {
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
            TypeContext::default(),
        )
        .expect("parsed lexicon")
    }

    #[test]
    fn find_variablizations_test() {
        let mut lex = create_test_lexicon();

        let rule = parse_rule(".(C .(.(CONS x_) NIL)) = .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) NIL))))))",
            &mut lex,
        )
            .expect("parsed rule");
        let opt = TRS::find_variablizations(&rule, &mut lex);
        assert!(opt.is_some());
        let vs = opt.unwrap();
        for (tp, places) in &vs {
            println!("{} {:?}\n\n", tp, places);
        }
        // list [[0, 1, 1], [1, 1, 1, 1, 1, 1, 1]]
        // list [[0, 1], [1, 1, 1, 1, 1, 1]]
        // nat → list → list [[0, 1, 0, 0], [1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]]
        // list → list [[0, 1, 0], [1, 0], [1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0]]
        // list → list [[0, 0]]
        assert_eq!(vs.len(), 5);

        let rule =
            parse_rule(".(C .(.(CONS .(DIGIT 9)) NIL)) = NIL", &mut lex).expect("parsed rule");
        let opt = TRS::find_variablizations(&rule, &mut lex);
        assert!(opt.is_some());
        let vs = opt.unwrap();
        for (tp, places) in &vs {
            println!("{} {:?}\n\n", tp, places);
        }
        // int [[0, 1, 0, 1, 1]]
        // nat [[0, 1, 0, 1]]
        // nat → list → list [[0, 1, 0, 0]]
        // int → nat [[0, 1, 0, 1, 0]]
        // list → list [[0, 1, 0]]
        // list [[0, 1, 1], [1]]
        // list [[0, 1]]
        // list → list [[0, 0]]
        assert_eq!(vs.len(), 8);

        let rule = parse_rule(
            ".(C .(.(CONS .(DIGIT 2)) var13_)) = .(.(CONS .(DIGIT 2)) .(C var13_))",
            &mut lex,
        )
        .expect("parsed rule");
        let opt = TRS::find_variablizations(&rule, &mut lex);
        assert!(opt.is_some());
        let vs = opt.unwrap();
        for (tp, places) in &vs {
            println!("{} {:?}", tp, places);
        }
        // list → list [[0, 1, 0], [1, 0]]
        // list → list [[0, 0], [1, 1, 0]]
        // int → nat [[0, 1, 0, 1, 0], [1, 0, 1, 0]]
        // nat → list → list [[0, 1, 0, 0], [1, 0, 0]]
        // list [[0, 1]]
        // nat [[0, 1, 0, 1], [1, 0, 1]]
        // int [[0, 1, 0, 1, 1], [1, 0, 1, 1]]
        assert_eq!(vs.len(), 7);
    }

    #[test]
    fn find_all_variablizations_test() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS .(DIGIT 9)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 2)) var13_)) = .(.(CONS .(DIGIT 2)) .(C var13_));", &mut lex, true, &[])
            .expect("parsed trs");
        let vs = trs.find_all_variablizations();

        for (tp, places) in &vs {
            println!("{} {:?}\n\n", tp, places);
        }
        // int [[0, 1, 0, 1, 1], [1, 0, 1, 1]]
        // int [[0, 1, 0, 1, 1]]
        // int → nat [[0, 1, 0, 1, 0], [1, 0, 1, 0]]
        // int → nat [[0, 1, 0, 1, 0]]
        // list [[0, 1, 1], [1]]
        // list [[0, 1]]
        // list → list [[0, 0], [1, 1, 0]]
        // list → list [[0, 0]]
        // list → list [[0, 1, 0], [1, 0]]
        // list → list [[0, 1, 0]]
        // nat [[0, 1, 0, 1], [1, 0, 1]]
        // nat [[0, 1, 0, 1]]
        // nat → list → list [[0, 1, 0, 0], [1, 0, 0]]
        // nat → list → list [[0, 1, 0, 0]]
        assert_eq!(vs.len(), 14);
    }

    #[test]
    fn analyze_variablizations_test() {
        let mut lex = create_test_lexicon();
        let mut trs = parse_trs(".(C .(.(CONS .(DIGIT 9)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 2)) var13_)) = .(.(CONS .(DIGIT 2)) .(C var13_));", &mut lex, true, &[])
            .expect("parsed trs");
        let vs = trs.find_all_variablizations();
        let xs = trs
            .analyze_variablizations(&vs)
            .into_iter()
            .map(|(x, y)| (y, x))
            .into_group_map();

        for (k, vs) in xs.iter() {
            for r in k {
                println!("{}", r.pretty(&lex.signature()));
            }
            for (tp, places) in vs {
                println!("  {} {:?}", tp, places);
            }
        }
        assert_eq!(xs.len(), 3);
        let mut lengths = xs.values().map(|v| v.len()).collect_vec();
        lengths.sort();
        assert_eq!(lengths, vec![2, 6, 6]);
    }

    #[test]
    fn apply_variablization_test_0() {
        let mut lex = create_test_lexicon();
        let mut trs = parse_trs(".(C .(.(CONS .(DIGIT 9)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 2)) x_)) = .(.(CONS .(DIGIT 2)) .(C x_));", &mut lex, true, &[])
            .expect("parsed trs");
        let vs = trs.find_all_variablizations();
        let xs = trs
            .analyze_variablizations(&vs)
            .into_iter()
            .map(|(x, y)| (y, x))
            .into_group_map();

        let mut trss = vec![];
        for (rs, vs) in xs.iter() {
            for (tp, places) in vs {
                println!("  {} {:?}", tp, places);
                if let Some(new_rs) = trs.apply_variablization(tp, places, rs) {
                    println!(
                        "{}\n",
                        new_rs
                            .iter()
                            .map(|r| r.pretty(&trs.lex.signature()))
                            .join("\n")
                    );
                    trss.push(new_rs);
                }
            }
        }
        assert_eq!(trss.len(), 14);
    }

    #[test]
    fn apply_variablization_test_1() {
        let mut lex = create_test_lexicon();
        let mut trs = parse_trs(".(C .(.(CONS .(DIGIT 9)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 2)) x_)) = .(.(CONS .(DIGIT 2)) .(C x_));  .(C .(.(CONS .(DIGIT 4)) x_)) = .(.(CONS .(DIGIT 4)) .(C x_));", &mut lex, true, &[]).expect("parsed trs");
        let vs = trs.find_all_variablizations();
        let xs = trs
            .analyze_variablizations(&vs)
            .into_iter()
            .map(|(x, y)| (y, x))
            .into_group_map();

        let mut trss = vec![];
        for (rs, vs) in xs.iter() {
            for (tp, places) in vs {
                println!("  {} {:?}", tp, places);
                if let Some(new_rs) = trs.apply_variablization(tp, places, rs) {
                    println!(
                        "{}\n",
                        new_rs
                            .iter()
                            .map(|r| r.pretty(&trs.lex.signature()))
                            .join("\n")
                    );
                    trss.push(new_rs);
                }
            }
        }
        assert_eq!(trss.len(), 14);
    }

    #[test]
    fn augment_trss_test() {
        let mut lex = create_test_lexicon();
        let mut trs = parse_trs(".(C .(.(CONS .(DIGIT 9)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 2)) x_)) = .(.(CONS .(DIGIT 2)) .(C x_));  .(C .(.(CONS .(DIGIT 4)) x_)) = .(.(CONS .(DIGIT 4)) .(C x_));", &mut lex, true, &[])
            .expect("parsed trs");
        let vs = trs.find_all_variablizations();
        let xs = trs
            .analyze_variablizations(&vs)
            .into_iter()
            .map(|(x, y)| (y, x))
            .into_group_map();

        let mut trss = vec![];
        for x in xs {
            trs.augment_trss(&mut trss, x);
        }
        for trs in &trss {
            println!("{}\n", trs);
        }
        assert_eq!(trss.len(), 16);
    }

    #[test]
    fn variablize_test_0() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(
            "C (CONS (DIGIT 0) (CONS (DIGIT 1) NIL)) = CONS (DIGIT 0) NIL;",
            &mut lex,
            true,
            &[],
        )
        .expect("parsed trs");

        let trss = trs.variablize();
        assert!(trss.is_ok());
        let trss = trss.unwrap();
        for trs in &trss {
            println!("{}\n", trs);
        }
        // .(C .(.(CONS .(DIGIT 0)) .(.(CONS .(DIGIT 1)) var0_))) = .(.(CONS .(DIGIT 0)) var0_);
        // .(var11_ .(var5_ .(var8_ var0_))) = .(var5_ var0_);
        // .(C .(.(CONS .(DIGIT var0_)) .(.(CONS .(DIGIT 1)) NIL))) = .(.(CONS .(DIGIT var0_)) NIL);
        // .(var10_ .(var4_ var8_)) = .(var4_ NIL);
        // .(C .(.(CONS .(var0_ 0)) .(.(CONS .(var0_ 1)) NIL))) = .(.(CONS .(var0_ 0)) NIL);
        // .(C .(.(CONS var0_) .(.(CONS .(DIGIT 1)) NIL))) = .(.(CONS var0_) NIL);
        // .(C .(.(var0_ .(DIGIT 0)) .(.(var0_ .(DIGIT 1)) NIL))) = .(.(var0_ .(DIGIT 0)) NIL);
        // .(C .(var0_ .(.(CONS .(DIGIT 1)) NIL))) = .(var0_ NIL);
        // .(C .(.(CONS .(DIGIT 0)) .(.(CONS .(DIGIT var0_)) NIL))) = .(.(CONS .(DIGIT 0)) NIL);
        // .(var5_ var4_) = .(.(CONS .(DIGIT 0)) NIL);
        // .(C .(.(CONS .(DIGIT 0)) .(.(CONS var0_) NIL))) = .(.(CONS .(DIGIT 0)) NIL);
        // .(C .(.(CONS .(DIGIT 0)) .(var0_ NIL))) = .(.(CONS .(DIGIT 0)) NIL);
        // .(C .(.(CONS .(DIGIT 0)) var0_)) = .(.(CONS .(DIGIT 0)) NIL);
        // .(C var0_) = .(.(CONS .(DIGIT 0)) NIL);
        // .(var0_ .(.(CONS .(DIGIT 0)) .(.(CONS .(DIGIT 1)) NIL))) = .(.(CONS .(DIGIT 0)) NIL);
        assert_eq!(trss.len(), 15);
    }

    #[test]
    fn variablize_test_1() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs("C (CONS (DIGIT 5) (CONS (DIGIT 3) (CONS (DIGIT 2) (CONS (DIGIT 9) (CONS (DIGIT 8) NIL))))) = CONS (DIGIT 2) NIL;C (x_ (x_ y_)) = y_;", &mut lex, true, &[]).expect("parsed trs");

        let trss = trs.variablize();
        assert!(trss.is_ok());
        let trss = trss.unwrap();
        for trs in &trss {
            println!("{}\n", trs);
        }
        assert_eq!(trss.len(), 26);
    }

    #[test]
    fn variablize_test_2() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS .(.(DECC .(DIGIT 5)) 4)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 0)) NIL)) = NIL;", &mut lex, true, &[]).expect("parsed trs");

        let vs = trs.find_all_variablizations();

        for (tp, places) in &vs {
            println!("{} {:?}\n\n", tp, places);
        }

        let trss = trs.variablize();

        assert!(trss.is_ok());

        let trss = trss.unwrap();
        for trs in &trss {
            println!("{}\n", trs);
        }

        assert_eq!(trss.len(), 15);
    }

    #[test]
    fn variablize_test_3() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS .(DIGIT 2)) var13_)) = .(.(CONS .(DIGIT 2)) .(C var13_)); .(C .(.(CONS .(DIGIT 0)) var20_)) = .(.(CONS .(DIGIT 0)) .(C var20_)); .(C .(.(CONS .(.(DECC .(DIGIT 7)) 7)) var19_)) = .(.(CONS .(.(DECC .(DIGIT 7)) 7)) .(C var19_)); .(C .(.(CONS .(.(DECC .(DIGIT 2)) 0)) var17_)) = .(.(CONS .(.(DECC .(DIGIT 2)) 0)) .(C var17_)); .(C .(.(CONS .(DIGIT 9)) var16_)) = .(.(CONS .(DIGIT 9)) .(C var16_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 0)) var15_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 0)) .(C var15_)); .(C .(.(CONS .(DIGIT 3)) var14_)) = .(.(CONS .(DIGIT 3)) .(C var14_)); .(C .(.(CONS .(.(DECC .(DIGIT 3)) 2)) var23_)) = .(.(CONS .(.(DECC .(DIGIT 3)) 2)) .(C var23_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 6)) var22_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 6)) .(C var22_));", &mut lex, true, &[]).expect("parsed trs");

        let vs = trs.find_all_variablizations();

        for (tp, places) in &vs {
            println!("{} {:?}\n\n", tp, places);
        }

        let trss = trs.variablize();

        assert!(trss.is_ok());

        let trss = trss.unwrap();
        for trs in &trss {
            println!("{}\n", trs);
        }

        assert_eq!(trss.len(), 13);
    }

    #[test]
    fn variablize_test_4() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS .(.(DECC .(DIGIT 5)) 4)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 0)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 2)) var13_)) = .(.(CONS .(DIGIT 2)) .(C var13_)); .(C .(.(CONS .(DIGIT 0)) var20_)) = .(.(CONS .(DIGIT 0)) .(C var20_)); .(C .(.(CONS .(.(DECC .(DIGIT 7)) 7)) var19_)) = .(.(CONS .(.(DECC .(DIGIT 7)) 7)) .(C var19_)); .(C .(.(CONS .(.(DECC .(DIGIT 2)) 0)) var17_)) = .(.(CONS .(.(DECC .(DIGIT 2)) 0)) .(C var17_)); .(C .(.(CONS .(DIGIT 9)) var16_)) = .(.(CONS .(DIGIT 9)) .(C var16_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 0)) var15_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 0)) .(C var15_)); .(C .(.(CONS .(DIGIT 3)) var14_)) = .(.(CONS .(DIGIT 3)) .(C var14_)); .(C .(.(CONS .(.(DECC .(DIGIT 3)) 2)) var23_)) = .(.(CONS .(.(DECC .(DIGIT 3)) 2)) .(C var23_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 6)) var22_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 6)) .(C var22_));", &mut lex, true, &[]).expect("parsed trs");

        let vs = trs.find_all_variablizations();

        for (tp, places) in &vs {
            println!("{} {:?}\n\n", tp, places);
        }

        let trss = trs.variablize();

        assert!(trss.is_ok());

        let trss = trss.unwrap();
        for trs in &trss {
            println!("{}\n", trs);
        }

        assert_eq!(trss.len(), 26);
    }
}

use super::{super::as_result, Lexicon, SampleError, TRS};
use itertools::Itertools;
use polytype::Type;
use std::collections::HashMap;
use term_rewriting::{Place, Rule, Term};

type Variablization = (Type, Vec<Place>);
type Transformation<'a> = (Rule, &'a Rule);
type AffectedRules<'a> = Vec<&'a Rule>;
type NewRules = Vec<Rule>;
type Cluster<'a> = (AffectedRules<'a>, Vec<(&'a Variablization, NewRules)>);

impl TRS {
    /// Replace subterms of [`term_rewriting::Rule`]s with [`term_rewriting::Variable`]s.
    /// [`term_rewriting::Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
    /// [`term_rewriting::Variable`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Variable.html
    pub fn variablize(&self) -> Result<Vec<TRS>, SampleError> {
        let snapshot = self.lex.snapshot();
        let clauses = self.clauses_for_learning(&[])?;
        let mut trss = vec![];
        // Generate the list of variablizations.
        let variablizations = TRS::find_all_variablizations(&clauses, &self.lex);
        // Apply variablizations and record which rules are affected.
        let new_ruless = TRS::apply_variablizations(&variablizations, &clauses, &self.lex);
        // Group the results by the affected rules.
        new_ruless
            .into_iter()
            .map(|(v, rs)| {
                let (nrs, ors) = rs.into_iter().unzip();
                (ors, (v, nrs))
            })
            .into_group_map()
            .into_iter()
            .for_each(|(old_rules, mut vars): Cluster| {
                // Sort variablizations by the lexicographically deepest place they affect.
                vars.sort_by_key(|((_, places), _)| places.iter().max().unwrap());
                vars.reverse();
                // For each variablization, v:
                for (i, (_, nrs)) in vars.iter().enumerate() {
                    let mut new_rules = nrs.clone();
                    let the_len = trss.len();
                    self.store_solution(&mut trss, TRS::make_solution(&new_rules, &old_rules));
                    if trss.len() == the_len {
                        continue;
                    }
                    // For each remaining variablization, v':
                    for ((tp, places), _) in vars.iter().skip(i + 1) {
                        // If you can apply v' to each rule, update the result.
                        let attempt = TRS::apply_variablization(tp, places, &new_rules, &self.lex);
                        if attempt.len() == new_rules.len() {
                            new_rules = attempt.into_iter().map(|(x, _)| x).collect_vec();
                        }
                    }
                    self.store_solution(&mut trss, TRS::make_solution(&new_rules, &old_rules));
                }
            });
        // Return the solution set.
        self.lex.rollback(snapshot);
        as_result(trss)
    }
    /// A simplified version of [`TRS::variablize`] that introduces exactly one
    /// [`term_rewriting::Variable`].
    ///
    /// [`term_rewriting::Variable`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Variable.html
    pub fn variablize_once(&self) -> Result<Vec<TRS>, SampleError> {
        let snapshot = self.lex.snapshot();
        let clauses = self.clauses_for_learning(&[])?;
        let variablizations = TRS::find_all_variablizations(&clauses, &self.lex);
        let mut trss = vec![];
        TRS::apply_variablizations(&variablizations, &clauses, &self.lex)
            .into_iter()
            .map(|(_, x)| x)
            .for_each(|solution| self.store_solution(&mut trss, solution));
        self.lex.rollback(snapshot);
        as_result(trss)
    }
    fn find_all_variablizations(rules: &[Rule], lex: &Lexicon) -> Vec<(Type, Vec<Place>)> {
        rules
            .iter()
            .filter_map(|rule| TRS::find_variablizations(rule, lex))
            .flatten()
            .unique()
            .collect_vec()
    }
    fn find_variablizations(rule: &Rule, lex: &Lexicon) -> Option<Vec<(Type, Vec<Place>)>> {
        // TODO: Using type *equality* rather than unification could reduce the
        // scope of variablization. Not a problem yet.
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
            .map(|((_, tp), places)| (tp, places))
            .collect_vec();
        Some(map)
    }
    fn apply_variablizations<'a, 'b>(
        vs: &'b [(Type, Vec<Place>)],
        clauses: &'a [Rule],
        lex: &Lexicon,
    ) -> Vec<(&'b Variablization, Vec<Transformation<'a>>)> {
        vs.iter()
            .map(|v| (v, TRS::apply_variablization(&v.0, &v.1, clauses, lex)))
            .filter(|(_, x)| !x.is_empty())
            .collect_vec()
    }
    fn apply_variablization<'a>(
        tp: &Type,
        places: &[Place],
        clauses: &'a [Rule],
        lex: &Lexicon,
    ) -> Vec<Transformation<'a>> {
        clauses
            .iter()
            .filter_map(|rule| {
                TRS::apply_variablization_to_rule(tp, places, rule, lex).map(|x| (x, rule))
            })
            .collect_vec()
    }
    fn apply_variablization_to_rule(
        tp: &Type,
        places: &[Place],
        rule: &Rule,
        lex: &Lexicon,
    ) -> Option<Rule> {
        let mut types = HashMap::new();
        lex.infer_rule(&rule, &mut types).drop().ok()?;
        places
            .get(0)
            .and_then(|place| rule.at(place))
            .and_then(|term| {
                if places
                    .iter()
                    .all(|place| types.get(place) == Some(tp) && rule.at(place) == Some(term))
                {
                    rule.replace_all(places, Term::Variable(lex.invent_variable(tp)))
                } else {
                    None
                }
            })
    }
    fn make_solution<'a>(new_rules: &[Rule], old_rules: &'a [&Rule]) -> Vec<(Rule, &'a Rule)> {
        new_rules
            .iter()
            .cloned()
            .zip(old_rules.iter().copied())
            .collect_vec()
    }
    fn store_solution(&self, new_trss: &mut Vec<TRS>, solution: Vec<(Rule, &Rule)>) {
        if let Some(trs) = self.adopt_solution(solution) {
            if new_trss.iter().all(|new_trs| !TRS::is_alpha(new_trs, &trs)) {
                new_trss.push(trs);
            }
        }
    }
    fn adopt_solution(&self, mut rules: Vec<(Rule, &Rule)>) -> Option<TRS> {
        let mut trs = self.clone();
        let mut i = 0;
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
        if self.lex.is_deterministic() {
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
        if rules.is_empty() {
            None
        } else {
            trs.swap_rules(&rules).ok()?;
            trs.smart_delete(0, 0).ok()
        }
    }
}

#[cfg(test)]
mod tests {
    use polytype::Context as TypeContext;
    use trs::parser::{parse_lexicon, parse_rule, parse_trs};
    use trs::{Lexicon, TRS};

    fn create_test_lexicon() -> Lexicon {
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
            "",
            "",
            true,
            TypeContext::default(),
        )
        .expect("parsed lexicon")
    }

    #[test]
    fn find_variablizations_test() {
        let lex = create_test_lexicon();
        let rule = parse_rule(".(C .(.(CONS x_) NIL)) = .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) NIL))))))",
            &lex,
        )
            .expect("parsed trs");
        let opt = TRS::find_variablizations(&rule, &lex);
        assert!(opt.is_some());
        let variablizations = opt.unwrap();
        let vs = variablizations.iter().collect::<Vec<_>>();

        for (tp, places) in &vs {
            println!("{} {:?}\n\n", tp, places);
        }

        assert_eq!(vs.len(), 6);
    }

    #[test]
    fn variablize_once_test_1() {
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
        let trss = trs.variablize_once().unwrap();

        for trs in &trss {
            println!("{}\n", trs);
        }

        assert_eq!(trss.len(), 4);
    }

    #[test]
    fn variablize_once_test_2() {
        let lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS x_) NIL)) = .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) .(.(CONS x_) NIL))))));",
            &lex,
        )
            .expect("parsed trs");
        let trss = trs.variablize_once().unwrap();

        for trs in &trss {
            println!("{}\n", trs);
        }

        assert_eq!(trss.len(), 4);
    }

    #[test]
    fn variablize_once_test_3() {
        let lex = create_test_lexicon();
        let trs = parse_trs(
            ".(C .(x_ NIL)) = .(x_ .(x_ .(x_ .(x_ .(x_ .(x_ NIL))))));",
            &lex,
        )
        .expect("parsed rule");
        let trss = trs.variablize_once().unwrap();

        for trs in &trss {
            println!("{}\n", trs);
        }

        assert_eq!(trss.len(), 2);
    }

    #[test]
    fn variablize_test_1() {
        let lex = create_test_lexicon();
        let trs = parse_trs("C (CONS (DIGIT 5) (CONS (DIGIT 3) (CONS (DIGIT 2) (CONS (DIGIT 9) (CONS (DIGIT 8) NIL))))) = CONS (DIGIT 2) NIL;C (x_ (x_ y_)) = y_;", &lex).expect("parsed trs");

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
        let lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS .(.(DECC .(DIGIT 5)) 4)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 0)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 2)) var13_)) = .(.(CONS .(DIGIT 2)) .(C var13_)); .(C .(.(CONS .(DIGIT 0)) var20_)) = .(.(CONS .(DIGIT 0)) .(C var20_)); .(C .(.(CONS .(.(DECC .(DIGIT 7)) 7)) var19_)) = .(.(CONS .(.(DECC .(DIGIT 7)) 7)) .(C var19_)); .(C .(.(CONS .(.(DECC .(DIGIT 2)) 0)) var17_)) = .(.(CONS .(.(DECC .(DIGIT 2)) 0)) .(C var17_)); .(C .(.(CONS .(DIGIT 9)) var16_)) = .(.(CONS .(DIGIT 9)) .(C var16_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 0)) var15_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 0)) .(C var15_)); .(C .(.(CONS .(DIGIT 3)) var14_)) = .(.(CONS .(DIGIT 3)) .(C var14_)); .(C .(.(CONS .(.(DECC .(DIGIT 3)) 2)) var23_)) = .(.(CONS .(.(DECC .(DIGIT 3)) 2)) .(C var23_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 6)) var22_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 6)) .(C var22_));", &lex).expect("parsed trs");

        let vs = TRS::find_all_variablizations(&trs.clauses_for_learning(&[]).unwrap(), &lex);

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

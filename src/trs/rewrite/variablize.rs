use super::{super::as_result, SampleError, TRS};
use itertools::Itertools;
use polytype::Type;
use std::{collections::HashMap, convert::TryFrom};
use term_rewriting::{Context, Place, Rule, RuleContext, Term, Variable};

pub type Variablization = (usize, Type, Vec<Place>);
//type Rules = Vec<usize>;
//type Cluster<'a> = Vec<(Rules, &'a Variablization)>;
type Types = HashMap<Rule, HashMap<Place, Type>>;

impl<'a, 'b> TRS<'a, 'b> {
    /// Compress the `TRS` by computing least general generalizations of its rules.
    pub fn lgg(&self) -> Result<TRS<'a, 'b>, SampleError> {
        if self.len() < 2 {
            return Err(SampleError::Subterm);
        }
        let mut l2 = self.utrs.clauses();
        l2.reverse();
        let mut l1 = Vec::with_capacity(l2.len());
        l1.push(l2.pop().unwrap());
        'next_rule: for r2 in l2.into_iter().rev() {
            for r1 in &mut l1 {
                if let Some(r3) = Rule::least_general_generalization(r1, &r2) {
                    *r1 = r3;
                    continue 'next_rule;
                }
            }
            l1.push(r2);
        }
        for r1 in l1.iter_mut() {
            r1.canonicalize(&mut HashMap::new());
        }
        let mut trs = self.clone();
        trs.utrs.rules = l1;
        Ok(trs)
    }
    /// Replace subterms of [`term_rewriting::Rule`]s with [`term_rewriting::Variable`]s.
    ///
    /// [`term_rewriting::Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
    /// [`term_rewriting::Variable`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Variable.html
    pub fn variablize(&self) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        let trs = if self.len() < 2 {
            self.clone()
        } else {
            self.lgg()?
        };
        let mut trss = vec![];
        let mut types = trs.collect_types();
        let vars = trs.find_all_variablizations(&types);
        trs.augment_trss(&mut trss, vars, &mut types);
        as_result(trss)
    }
    pub fn analyze_variablizations_by_depth(&self) -> (Types, Vec<Variablization>) {
        let types = self.collect_types();
        let all_vars = self.find_all_variablizations(&types);
        (types, all_vars)
    }
    pub fn variablize_by(
        &self,
        (affected, tp, places): &Variablization,
        types: &mut Types,
    ) -> Option<TRS<'a, 'b>> {
        let clauses = self.utrs.clauses();
        let mut new_rules = self.apply_variablization(tp, places, *affected, &clauses, types)?;
        self.adopt_solution(&mut new_rules)
    }
    fn collect_types(&self) -> Types {
        self.clauses()
            .into_iter()
            .filter_map(|(_, r)| {
                let mut ctx = self.lex.0.ctx.clone();
                let mut types = HashMap::new();
                self.lex
                    .infer_rule(&r, &mut types, &mut ctx)
                    .ok()
                    .map(|_| (r, types))
            })
            .collect()
    }
    fn augment_trss(
        &self,
        trss: &mut Vec<TRS<'a, 'b>>,
        mut vars: Vec<Variablization>,
        types: &mut Types,
    ) {
        // Sort variablizations by the lexicographically deepest place they affect.
        let clauses = self.utrs.clauses();
        let self_len = clauses.len();
        vars.sort_by_key(|(rule, _, places)| {
            let best_place = places.iter().filter(|place| place[0] == 0).max().unwrap();
            (self_len - *rule, best_place.clone())
        });
        // if unique so far, push self onto trss
        if self.unique_shape(trss) {
            trss.push(self.clone())
        }
        // For each variablization:
        let mut new_trss = Vec::new();
        while let Some((affected, tp, places)) = vars.pop() {
            // For each trs in trss:
            for trs in trss.iter() {
                let clauses = trs.utrs.clauses();
                // Try the variablization
                if let Some(mut new_rules) =
                    trs.apply_variablization(&tp, &places, affected, &clauses, types)
                {
                    if let Some(new_trs) = trs.adopt_solution(&mut new_rules) {
                        // Add any unique ones to trss.
                        if new_trs.unique_shape(trss) && new_trs.unique_shape(&new_trss) {
                            new_trss.push(new_trs);
                        }
                    }
                }
            }
            trss.append(&mut new_trss);
        }
    }
    fn find_all_variablizations(&self, types: &Types) -> Vec<Variablization> {
        self.utrs
            .clauses()
            .iter()
            .enumerate()
            .filter_map(|(i, rule)| TRS::find_variablizations(i, rule, types))
            .flatten()
            .unique()
            .collect_vec()
    }
    fn find_variablizations(n: usize, rule: &Rule, types: &Types) -> Option<Vec<Variablization>> {
        // List the places where each term/type token occurs.
        let types = types.get(rule)?;
        let map = rule
            .subterms()
            .into_iter()
            .filter(|(term, _)| term.as_application().is_some())
            .map(|(term, place)| ((term, types.get(&place).unwrap().clone()), place))
            .into_group_map()
            .into_iter()
            .filter(|(_, places)| places.iter().any(|place| place[0] == 0))
            .filter(|(_, places)| !places.contains(&vec![0]))
            .map(|((_, tp), places)| (n, tp, places))
            .collect_vec();
        Some(map)
    }
    fn apply_variablization(
        &self,
        tp: &Type,
        places: &[Place],
        affected: usize,
        rules: &[Rule],
        types: &mut Types,
    ) -> Option<Vec<Rule>> {
        let mut new_rules = rules.to_vec();
        match self.apply_variablization_to_rule(tp, places, &rules[affected], types) {
            Some(new_rule) => new_rules[affected] = new_rule,
            None => return None,
        }
        Some(new_rules)
    }
    fn apply_variablization_to_rule(
        &self,
        tp: &Type,
        places: &[Place],
        rule: &Rule,
        types: &mut Types,
    ) -> Option<Rule> {
        places
            .get(0)
            .and_then(|place| rule.at(place))
            .and_then(|term| {
                let applies = {
                    let mut tp_rule = rule.clone();
                    tp_rule.canonicalize(&mut HashMap::new());
                    let tps = types.get(&tp_rule)?;
                    places
                        .iter()
                        .all(|place| tps.get(place) == Some(tp) && rule.at(place) == Some(term))
                };
                if applies {
                    let mut context =
                        RuleContext::from(rule.clone()).replace_all(places, Context::Hole)?;
                    context.canonicalize(&mut HashMap::new());
                    let id = context.lhs.variables().len();
                    let context =
                        context.replace_all(places, Context::Variable(Variable { id }))?;
                    let mut new_rule = Rule::try_from(&context).ok()?;
                    new_rule.canonicalize(&mut HashMap::new());
                    if !types.contains_key(&new_rule) {
                        types.insert(new_rule.clone(), types.get(rule)?.clone());
                    }
                    Some(new_rule)
                } else {
                    None
                }
            })
    }
    fn adopt_solution(&self, rules: &mut Vec<Rule>) -> Option<TRS<'a, 'b>> {
        let self_len = self.len();

        self.filter_background(rules);
        if rules.len() != self_len {
            return None;
        }

        for i in 0..rules.len() {
            // Ensure alpha-unique rules.
            for j in 0..i {
                if Rule::alpha(&rules[i], &rules[j]).is_some() {
                    return None;
                }
            }
            // Ensure alpha-unique LHSs if deterministic.
            if self.is_deterministic() {
                for j in (i + 1)..rules.len() {
                    if Term::alpha(vec![(&rules[j].lhs, &rules[i].lhs)]).is_some() {
                        return None;
                    }
                }
            }
        }

        // Create a new TRS.
        let mut trs = self.clone();
        trs.utrs.rules = rules.to_vec();
        trs.smart_delete(0, 0).ok().and_then(|trs| {
            if trs.len() == self_len {
                Some(trs)
            } else {
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use polytype::Context as TypeContext;
    use std::collections::HashMap;
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
    fn find_variablizations_test_0() {
        let mut lex = create_test_lexicon();
        let rule =
            parse_rule(".(C .(.(CONS .(DIGIT 9)) NIL)) = NIL", &mut lex).expect("parsed rule");
        let mut types = HashMap::new();
        let mut types2 = HashMap::new();
        let mut ctx = lex.0.ctx.clone();
        lex.infer_rule(&rule, &mut types2, &mut ctx).unwrap();
        types.insert(rule.clone(), types2);
        let opt = TRS::find_variablizations(0, &rule, &types);
        assert!(opt.is_some());
        let vs = opt.unwrap();

        assert_eq!(
            vs.iter()
                .map(|(n, tp, places)| format!("{} {} {:?}", n, tp, places))
                .sorted()
                .join(" "),
            [
                "0 int [[0, 1, 0, 1, 1]]",
                "0 int → nat [[0, 1, 0, 1, 0]]",
                "0 list [[0, 1, 1], [1]]",
                "0 list [[0, 1]]",
                "0 list → list [[0, 0]]",
                "0 list → list [[0, 1, 0]]",
                "0 nat [[0, 1, 0, 1]]",
                "0 nat → list → list [[0, 1, 0, 0]]",
            ]
            .iter()
            .join(" ")
        );
    }

    #[test]
    fn find_variablizations_test_1() {
        let mut lex = create_test_lexicon();
        let rule = parse_rule(
            ".(C .(.(CONS .(DIGIT 2)) var13_)) = .(.(CONS .(DIGIT 2)) .(C var13_))",
            &mut lex,
        )
        .expect("parsed rule");
        let mut types = HashMap::new();
        let mut types2 = HashMap::new();
        let mut ctx = lex.0.ctx.clone();
        lex.infer_rule(&rule, &mut types2, &mut ctx).unwrap();
        types.insert(rule.clone(), types2);
        let opt = TRS::find_variablizations(0, &rule, &types);
        assert!(opt.is_some());
        let vs = opt.unwrap();

        assert_eq!(
            vs.iter()
                .map(|(n, tp, places)| format!("{} {} {:?}", n, tp, places))
                .sorted()
                .join(" "),
            [
                "0 int [[0, 1, 0, 1, 1], [1, 0, 1, 1]]",
                "0 int → nat [[0, 1, 0, 1, 0], [1, 0, 1, 0]]",
                "0 list [[0, 1]]",
                "0 list → list [[0, 0], [1, 1, 0]]",
                "0 list → list [[0, 1, 0], [1, 0]]",
                "0 nat [[0, 1, 0, 1], [1, 0, 1]]",
                "0 nat → list → list [[0, 1, 0, 0], [1, 0, 0]]",
            ]
            .iter()
            .join(" ")
        );
    }

    #[test]
    fn find_all_variablizations_test() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS .(v0_ v1_)) NIL)) = NIL; .(C .(.(CONS .(v0_ v1_)) v2_)) = .(.(CONS .(v0_ v1_)) .(C v2_));", &mut lex, true, &[])
            .expect("parsed trs");
        let types = trs.collect_types();
        let vs = trs.find_all_variablizations(&types);

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
    }

    #[test]
    fn apply_variablization_test() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS .(v0_ v1_)) NIL)) = NIL; .(C .(.(CONS .(v0_ v1_)) v2_)) = .(.(CONS .(v0_ v1_)) .(C v2_));", &mut lex, true, &[])
            .expect("parsed trs");
        let mut types = trs.collect_types();
        let vs = trs.find_all_variablizations(&types);
        let clauses = trs.clauses().into_iter().map(|(_, r)| r).collect_vec();
        let mut result_strings = vec![];
        for (n, tp, places) in vs
            .into_iter()
            .sorted_by_key(|(n, tp, places)| format!("{} {} {:?}", n, tp, places))
        {
            result_strings.push(format!("{} {} {:?}", n, tp, places));
            if let Some(new_rs) = trs.apply_variablization(&tp, &places, n, &clauses, &mut types) {
                result_strings.push(format!(
                    "{}",
                    new_rs
                        .iter()
                        .map(|r| r.pretty(&trs.lex.signature()))
                        .join("; ")
                ));
            }
        }
        assert_eq!(
            result_strings.iter().join("\n"),
            [
                "0 list [[0, 1, 1], [1]]",
                "C (CONS (v0_ v1_) v2_) = v2_; C (CONS (v0_ v1_) v2_) = CONS (v0_ v1_) (C v2_)",
                "0 list [[0, 1]]",
                "C v0_ = []; C (CONS (v0_ v1_) v2_) = CONS (v0_ v1_) (C v2_)",
                "0 list → list [[0, 0]]",
                "v0_ (CONS (v1_ v2_) []) = []; C (CONS (v0_ v1_) v2_) = CONS (v0_ v1_) (C v2_)",
                "0 list → list [[0, 1, 0]]",
                "C (v0_ []) = []; C (CONS (v0_ v1_) v2_) = CONS (v0_ v1_) (C v2_)",
                "0 nat [[0, 1, 0, 1]]",
                "C (CONS v0_ []) = []; C (CONS (v0_ v1_) v2_) = CONS (v0_ v1_) (C v2_)",
                "0 nat → list → list [[0, 1, 0, 0]]",
                "C (v0_ (v1_ v2_) []) = []; C (CONS (v0_ v1_) v2_) = CONS (v0_ v1_) (C v2_)",
                "1 list [[0, 1]]",
                "1 list → list [[0, 0], [1, 1, 0]]",
                "C (CONS (v0_ v1_) []) = []; v0_ (CONS (v1_ v2_) v3_) = CONS (v1_ v2_) (v0_ v3_)",
                "1 list → list [[0, 1, 0], [1, 0]]",
                "C (CONS (v0_ v1_) []) = []; C (v0_ v1_) = v0_ (C v1_)",
                "1 nat [[0, 1, 0, 1], [1, 0, 1]]",
                "C (CONS (v0_ v1_) []) = []; C (CONS v0_ v1_) = CONS v0_ (C v1_)",
                "1 nat → list → list [[0, 1, 0, 0], [1, 0, 0]]",
                "C (CONS (v0_ v1_) []) = []; C (v0_ (v1_ v2_) v3_) = v0_ (v1_ v2_) (C v3_)",
            ]
            .iter()
            .join("\n")
        );
    }

    #[test]
    fn augment_trss_test_0() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS .(v0_ v1_)) NIL)) = NIL;", &mut lex, true, &[])
            .expect("parsed trs");
        let mut types = trs.collect_types();
        let mut trss = vec![];
        let vars = trs.find_all_variablizations(&types);
        trs.augment_trss(&mut trss, vars, &mut types);

        assert_eq!(
            trss.iter().map(|trs| trs.to_string()).join("\n"),
            [
                ".(C .(.(CONS .(v0_ v1_)) NIL)) = NIL;",
                ".(C .(.(CONS .(v0_ v1_)) v2_)) = v2_;",
                ".(C .(.(CONS v0_) NIL)) = NIL;",
                ".(C .(.(CONS v0_) v1_)) = v1_;",
                ".(C .(.(v0_ .(v1_ v2_)) NIL)) = NIL;",
                ".(C .(.(v0_ .(v1_ v2_)) v3_)) = v3_;",
                ".(C .(.(v0_ v1_) NIL)) = NIL;",
                ".(C .(.(v0_ v1_) v2_)) = v2_;",
                ".(C .(v0_ NIL)) = NIL;",
                ".(C .(v0_ v1_)) = v1_;",
                ".(C v0_) = NIL;",
                ".(v0_ .(.(CONS .(v1_ v2_)) NIL)) = NIL;",
                ".(v0_ .(.(CONS .(v1_ v2_)) v3_)) = v3_;",
                ".(v0_ .(.(CONS v1_) NIL)) = NIL;",
                ".(v0_ .(.(CONS v1_) v2_)) = v2_;",
                ".(v0_ .(.(v1_ .(v2_ v3_)) NIL)) = NIL;",
                ".(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = v4_;",
                ".(v0_ .(.(v1_ v2_) NIL)) = NIL;",
                ".(v0_ .(.(v1_ v2_) v3_)) = v3_;",
                ".(v0_ .(v1_ NIL)) = NIL;",
                ".(v0_ .(v1_ v2_)) = v2_;",
                ".(v0_ v1_) = NIL;",
            ]
            .iter()
            .join("\n")
        );
    }

    #[test]
    fn augment_trss_test_1() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(
            ".(C .(.(CONS .(v0_ v1_)) v2_)) = .(.(CONS .(v0_ v1_)) .(C v2_));",
            &mut lex,
            true,
            &[],
        )
        .expect("parsed trs");
        let mut types = trs.collect_types();
        let mut trss = vec![];
        let vars = trs.find_all_variablizations(&types);
        trs.augment_trss(&mut trss, vars, &mut types);

        assert_eq!(
            trss.iter().map(|trs| trs.to_string()).join("\n"),
            [
                ".(C .(.(CONS .(v0_ v1_)) v2_)) = .(.(CONS .(v0_ v1_)) .(C v2_));",
                ".(C .(.(CONS v0_) v1_)) = .(.(CONS v0_) .(C v1_));",
                ".(C .(.(v0_ .(v1_ v2_)) v3_)) = .(.(v0_ .(v1_ v2_)) .(C v3_));",
                ".(C .(.(v0_ v1_) v2_)) = .(.(v0_ v1_) .(C v2_));",
                ".(C .(v0_ v1_)) = .(v0_ .(C v1_));",
                ".(v0_ .(.(CONS .(v1_ v2_)) v3_)) = .(.(CONS .(v1_ v2_)) .(v0_ v3_));",
                ".(v0_ .(.(CONS v1_) v2_)) = .(.(CONS v1_) .(v0_ v2_));",
                ".(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = .(.(v1_ .(v2_ v3_)) .(v0_ v4_));",
                ".(v0_ .(.(v1_ v2_) v3_)) = .(.(v1_ v2_) .(v0_ v3_));",
                ".(v0_ .(v1_ v2_)) = .(v1_ .(v0_ v2_));",
            ]
            .iter()
            .join("\n")
        );
    }

    #[test]
    fn augment_trss_test_2() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS .(v0_ v1_)) NIL)) = NIL; .(C .(.(CONS .(v0_ v1_)) v2_)) = .(.(CONS .(v0_ v1_)) .(C v2_));", &mut lex, true, &[])
            .expect("parsed trs");
        let mut types = trs.collect_types();
        let mut trss = vec![];
        let vars = trs.find_all_variablizations(&types);
        trs.augment_trss(&mut trss, vars, &mut types);

        let rule_1s = vec![
            ".(C .(.(CONS .(v0_ v1_)) NIL)) = NIL;",
            ".(C .(.(CONS v0_) NIL)) = NIL;",
            ".(C .(.(v0_ .(v1_ v2_)) NIL)) = NIL;",
            ".(C .(.(v0_ v1_) NIL)) = NIL;",
            ".(C .(v0_ NIL)) = NIL;",
            ".(v0_ .(.(CONS .(v1_ v2_)) NIL)) = NIL;",
            ".(v0_ .(.(CONS v1_) NIL)) = NIL;",
            ".(v0_ .(.(v1_ .(v2_ v3_)) NIL)) = NIL;",
            ".(v0_ .(.(v1_ v2_) NIL)) = NIL;",
            ".(v0_ .(v1_ NIL)) = NIL;",
        ];

        let rule_2s = vec![
            ".(C .(.(CONS .(v0_ v1_)) v2_)) = .(.(CONS .(v0_ v1_)) .(C v2_));",
            ".(C .(.(CONS v0_) v1_)) = .(.(CONS v0_) .(C v1_));",
            ".(C .(.(v0_ .(v1_ v2_)) v3_)) = .(.(v0_ .(v1_ v2_)) .(C v3_));",
            ".(C .(.(v0_ v1_) v2_)) = .(.(v0_ v1_) .(C v2_));",
            ".(C .(v0_ v1_)) = .(v0_ .(C v1_));",
            ".(v0_ .(.(CONS .(v1_ v2_)) v3_)) = .(.(CONS .(v1_ v2_)) .(v0_ v3_));",
            ".(v0_ .(.(CONS v1_) v2_)) = .(.(CONS v1_) .(v0_ v2_));",
            ".(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = .(.(v1_ .(v2_ v3_)) .(v0_ v4_));",
            ".(v0_ .(.(v1_ v2_) v3_)) = .(.(v1_ v2_) .(v0_ v3_));",
            ".(v0_ .(v1_ v2_)) = .(v1_ .(v0_ v2_));",
        ];

        let trs_strs = trss.iter().map(|trs| trs.to_string()).collect_vec();
        for rule1 in &rule_1s {
            for rule2 in &rule_2s {
                assert!(trs_strs.contains(&format!("{}\n{}", rule1, rule2)));
            }
        }
    }

    #[test]
    fn lgg_test() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS .(.(DECC .(DIGIT 5)) 4)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 0)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 2)) var13_)) = .(.(CONS .(DIGIT 2)) .(C var13_)); .(C .(.(CONS .(DIGIT 0)) var20_)) = .(.(CONS .(DIGIT 0)) .(C var20_)); .(C .(.(CONS .(.(DECC .(DIGIT 7)) 7)) var19_)) = .(.(CONS .(.(DECC .(DIGIT 7)) 7)) .(C var19_)); .(C .(.(CONS .(.(DECC .(DIGIT 2)) 0)) var17_)) = .(.(CONS .(.(DECC .(DIGIT 2)) 0)) .(C var17_)); .(C .(.(CONS .(DIGIT 9)) var16_)) = .(.(CONS .(DIGIT 9)) .(C var16_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 0)) var15_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 0)) .(C var15_)); .(C .(.(CONS .(DIGIT 3)) var14_)) = .(.(CONS .(DIGIT 3)) .(C var14_)); .(C .(.(CONS .(.(DECC .(DIGIT 3)) 2)) var23_)) = .(.(CONS .(.(DECC .(DIGIT 3)) 2)) .(C var23_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 6)) var22_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 6)) .(C var22_));", &mut lex, true, &[]).expect("parsed trs");

        let maybe_trs = trs.lgg();
        assert!(maybe_trs.is_ok());

        let new_trs = maybe_trs.unwrap();
        println!("{}\n", new_trs);
        assert_eq!(new_trs.to_string(), ".(C .(.(CONS .(v0_ v1_)) NIL)) = NIL;\n.(C .(.(CONS .(v0_ v1_)) v2_)) = .(.(CONS .(v0_ v1_)) .(C v2_));");
    }

    #[test]
    fn variablize_test() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS .(.(DECC .(DIGIT 5)) 4)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 0)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 2)) var13_)) = .(.(CONS .(DIGIT 2)) .(C var13_)); .(C .(.(CONS .(DIGIT 0)) var20_)) = .(.(CONS .(DIGIT 0)) .(C var20_)); .(C .(.(CONS .(.(DECC .(DIGIT 7)) 7)) var19_)) = .(.(CONS .(.(DECC .(DIGIT 7)) 7)) .(C var19_)); .(C .(.(CONS .(.(DECC .(DIGIT 2)) 0)) var17_)) = .(.(CONS .(.(DECC .(DIGIT 2)) 0)) .(C var17_)); .(C .(.(CONS .(DIGIT 9)) var16_)) = .(.(CONS .(DIGIT 9)) .(C var16_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 0)) var15_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 0)) .(C var15_)); .(C .(.(CONS .(DIGIT 3)) var14_)) = .(.(CONS .(DIGIT 3)) .(C var14_)); .(C .(.(CONS .(.(DECC .(DIGIT 3)) 2)) var23_)) = .(.(CONS .(.(DECC .(DIGIT 3)) 2)) .(C var23_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 6)) var22_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 6)) .(C var22_));", &mut lex, true, &[]).expect("parsed trs");

        let trss = trs.variablize();
        assert_eq!(trss.unwrap().len(), 100);
    }
}

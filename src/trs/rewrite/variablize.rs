use itertools::Itertools;
use polytype::Type;
use std::{collections::HashMap, convert::TryFrom};
use term_rewriting::{Context, Place, Rule, RuleContext, Term, Variable};
use trs::{as_result, Environment, SampleError, TRS};

pub type Variablization = (usize, Type, Vec<Place>);
pub type Types = HashMap<Rule, HashMap<Place, Type>>;

impl<'a, 'b> TRS<'a, 'b> {
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
        let (ruless, combos) = trs.analyze_variablizations();
        let trss = combos
            .into_iter()
            .map(|combo| {
                let mut trs = trs.clone();
                for (m, n) in combo.into_iter().enumerate() {
                    trs.utrs.rules[m] = ruless[m][n].clone();
                }
                trs
            })
            .collect_vec();
        as_result(trss)
    }
    pub fn analyze_variablizations(&self) -> (Vec<Vec<Rule>>, Vec<Vec<usize>>) {
        if self.is_empty() {
            (vec![], vec![])
        } else {
            let trs = if self.len() < 2 {
                self.clone()
            } else {
                self.lgg().unwrap_or_else(|_| self.clone())
            };
            let mut types = trs.collect_types();
            let mut vars = trs.find_all_variablizations(&types);
            let new_rules = trs.compute_unique_rules(&mut vars, &mut types);
            let new_trss = trs.compute_unique_trss(&new_rules);
            (new_rules, new_trss)
        }
    }
    fn compute_unique_rules(
        &self,
        vars: &mut Vec<Variablization>,
        types: &mut Types,
    ) -> Vec<Vec<Rule>> {
        // Sort variablizations by the lexicographically deepest place they affect.
        let clauses = self.utrs.clauses();
        let self_len = clauses.len();
        vars.sort_by_key(|(rule, _, places)| {
            let best_place = places.iter().filter(|place| place[0] == 0).max().unwrap();
            (self_len - *rule, best_place.clone())
        });
        let mut all_rules = Vec::with_capacity(self.len());
        let mut n = 0;
        // Add the rule to the list.
        let mut rules = Vec::with_capacity(vars.len());
        rules.push(self.utrs.rules[0].clone());
        while let Some((affected, tp, places)) = vars.pop() {
            // Store the solutions whenever you move to the next rule.
            if affected != n {
                self.filter_background(&mut rules);
                all_rules.push(rules.clone());
                rules.clear();
                n = affected;
                rules.push(self.utrs.rules[n].clone());
            }
            let mut new_rules = Vec::with_capacity(rules.len());
            for rule in &rules {
                // Try the variablization
                if let Some(new_rule) = self.apply_variablization(&tp, &places, rule, types) {
                    // Keep unique ones.
                    if !new_rules
                        .iter()
                        .any(|other| Rule::same_shape(&new_rule, other))
                        && !rules.iter().any(|other| Rule::same_shape(&new_rule, other))
                    {
                        new_rules.push(new_rule);
                    }
                }
            }
            rules.append(&mut new_rules);
        }
        self.filter_background(&mut rules);
        all_rules.push(rules);
        all_rules
    }
    fn compute_unique_trss(&self, ruless: &[Vec<Rule>]) -> Vec<Vec<usize>> {
        let mut stack = vec![vec![]];
        let mut second_stack = vec![];
        for rules in ruless {
            while let Some(combo) = stack.pop() {
                for n in 0..rules.len() {
                    if self.rule_can_be_added(ruless, &combo, n) {
                        let mut new_combo = combo.clone();
                        new_combo.push(n);
                        second_stack.push(new_combo);
                    }
                }
            }
            std::mem::swap(&mut stack, &mut second_stack);
        }
        stack
    }
    fn rule_can_be_added(&self, ruless: &[Vec<Rule>], combo: &[usize], n: usize) -> bool {
        // TODO: incorrect for non-deterministic case
        let mut max = 0;
        for (m, &n) in combo.iter().enumerate() {
            if let Some(rule_max) = ruless[m][n].variables().iter().map(|v| v.id).max() {
                max = max.max(rule_max + 1);
            }
        }
        let bg_max = self
            .background
            .iter()
            .flat_map(|r| r.variables())
            .map(|v| v.id)
            .max()
            .map(|n| n + 1)
            .unwrap_or(0);
        max = max.max(bg_max);
        let mut lhs = ruless[combo.len()][n].lhs.clone();
        lhs.offset(max);
        let bg_lhss = self.background.iter().map(|r| &r.lhs);
        let combo_lhss = combo.iter().enumerate().map(|(m, &n)| &ruless[m][n].lhs);
        bg_lhss
            .chain(combo_lhss)
            .all(|prior| Term::pmatch(vec![(prior, &lhs)]).is_none())
    }
    pub fn variablize_by(
        &self,
        (affected, tp, places): &Variablization,
        types: &mut Types,
    ) -> Option<TRS<'a, 'b>> {
        let mut clauses = self.utrs.clauses();
        let rule = &clauses[*affected];
        let new_rule = self.apply_variablization(tp, places, rule, types)?;
        clauses[*affected] = new_rule;
        self.adopt_solution(&mut clauses)
    }
    pub fn collect_types(&self) -> Types {
        self.clauses()
            .into_iter()
            .filter_map(|(_, r)| {
                let mut ctx = self.lex.0.ctx.clone();
                let mut types = HashMap::new();
                let mut env = Environment::from_vars(&r.variables(), &mut ctx);
                self.lex
                    .infer_rule(&r, &mut types, &mut env, &mut ctx)
                    .ok()
                    .map(|_| (r, types))
            })
            .collect()
    }
    pub fn try_all_variablizations(&self) -> Vec<Vec<Rule>> {
        let mut types = self.collect_types();
        let vars = self.find_all_variablizations(&types);
        let clauses = self.utrs.clauses();
        clauses
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                vars.iter()
                    .filter(|(n, _, _)| i == *n)
                    .filter_map(|(_, tp, places)| {
                        self.apply_variablization(tp, places, &c, &mut types)
                    })
                    .collect_vec()
            })
            .collect_vec()
    }
    pub fn find_all_variablizations(&self, types: &Types) -> Vec<Variablization> {
        let clauses = self.utrs.clauses();
        let self_len = clauses.len();
        clauses
            .iter()
            .enumerate()
            .filter_map(|(i, rule)| TRS::find_variablizations(i, rule, types))
            .flatten()
            .unique()
            .sorted_by_key(|(rule, _, places)| {
                let best_place = places.iter().filter(|place| place[0] == 0).max().unwrap();
                (self_len - *rule, best_place.clone())
            })
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
    pub(crate) fn apply_variablization(
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
    pub(crate) fn adopt_solution(&self, rules: &mut Vec<Rule>) -> Option<TRS<'a, 'b>> {
        self.filter_background(rules);

        let mut i = 0;
        while i < rules.len() {
            if rules[..i]
                .iter()
                .any(|other| Rule::alpha(&other, &rules[i]).is_some())
            {
                rules.remove(i);
            } else if self.is_deterministic()
                && rules[..i]
                    .iter()
                    .any(|other| Term::alpha(vec![(&other.lhs, &rules[i].lhs)]).is_some())
            {
                return None;
            } else {
                i += 1;
            }
        }

        // Create a new TRS.
        let mut trs = self.clone();
        trs.utrs.rules = rules.to_vec();
        trs.smart_delete(0, 0).ok()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use polytype::Context as TypeContext;
    use std::collections::HashMap;
    use trs::parser::{parse_lexicon, parse_rule, parse_trs};
    use trs::{Environment, Lexicon, TRS};

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
        let mut env = Environment::from_vars(&rule.variables(), &mut ctx);
        lex.infer_rule(&rule, &mut types2, &mut env, &mut ctx)
            .unwrap();
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
        let mut env = Environment::from_vars(&rule.variables(), &mut ctx);
        lex.infer_rule(&rule, &mut types2, &mut env, &mut ctx)
            .unwrap();
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
    fn try_all_variablizations_test() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs("(C (CONS (DIGIT 2) (CONS (DIGIT 1) (CONS (DIGIT 5) (CONS (DIGIT 4) NIL))))) = (CONS (DIGIT 5) NIL);", &mut lex, true, &[])
            .expect("parsed trs");
        let ruless = trs.try_all_variablizations();
        for (i, rules) in ruless.iter().enumerate() {
            println!("{}", i);
            for rule in rules {
                println!("- {}", rule.pretty(lex.signature()));
            }
        }
        assert_eq!(ruless.len(), 1);
        assert_eq!(ruless[0].len(), 20);
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
            if let Some(new_r) = trs.apply_variablization(&tp, &places, &clauses[n], &mut types) {
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
    }

    #[test]
    fn compute_unique_rules_test() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS .(v0_ v1_)) NIL)) = NIL; .(C .(.(CONS .(v0_ v1_)) v2_)) = .(.(CONS .(v0_ v1_)) .(C v2_));", &mut lex, true, &[])
            .expect("parsed trs");
        let mut types = trs.collect_types();
        let mut vars = trs.find_all_variablizations(&types);
        let ruless = trs.compute_unique_rules(&mut vars, &mut types);
        // See augment_trss_test_[0,1]
        assert_eq!(ruless.len(), 2);

        assert_eq!(ruless[0].len(), 22);
        let firsts_expected = [
            ".(C .(.(CONS .(v0_ v1_)) NIL)) = NIL",
            ".(C .(.(CONS .(v0_ v1_)) v2_)) = v2_",
            ".(C .(.(CONS v0_) NIL)) = NIL",
            ".(C .(.(CONS v0_) v1_)) = v1_",
            ".(C .(.(v0_ .(v1_ v2_)) NIL)) = NIL",
            ".(C .(.(v0_ .(v1_ v2_)) v3_)) = v3_",
            ".(C .(.(v0_ v1_) NIL)) = NIL",
            ".(C .(.(v0_ v1_) v2_)) = v2_",
            ".(C .(v0_ NIL)) = NIL",
            ".(C .(v0_ v1_)) = v1_",
            ".(C v0_) = NIL",
            ".(v0_ .(.(CONS .(v1_ v2_)) NIL)) = NIL",
            ".(v0_ .(.(CONS .(v1_ v2_)) v3_)) = v3_",
            ".(v0_ .(.(CONS v1_) NIL)) = NIL",
            ".(v0_ .(.(CONS v1_) v2_)) = v2_",
            ".(v0_ .(.(v1_ .(v2_ v3_)) NIL)) = NIL",
            ".(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = v4_",
            ".(v0_ .(.(v1_ v2_) NIL)) = NIL",
            ".(v0_ .(.(v1_ v2_) v3_)) = v3_",
            ".(v0_ .(v1_ NIL)) = NIL",
            ".(v0_ .(v1_ v2_)) = v2_",
            ".(v0_ v1_) = NIL",
        ];
        for first in ruless[0].iter().map(|rule| rule.display(lex.signature())) {
            assert!(firsts_expected.contains(&first.as_str()));
        }

        assert_eq!(ruless[1].len(), 10);
        let seconds_expected = [
            ".(C .(.(CONS .(v0_ v1_)) v2_)) = .(.(CONS .(v0_ v1_)) .(C v2_))",
            ".(C .(.(CONS v0_) v1_)) = .(.(CONS v0_) .(C v1_))",
            ".(C .(.(v0_ .(v1_ v2_)) v3_)) = .(.(v0_ .(v1_ v2_)) .(C v3_))",
            ".(C .(.(v0_ v1_) v2_)) = .(.(v0_ v1_) .(C v2_))",
            ".(C .(v0_ v1_)) = .(v0_ .(C v1_))",
            ".(v0_ .(.(CONS .(v1_ v2_)) v3_)) = .(.(CONS .(v1_ v2_)) .(v0_ v3_))",
            ".(v0_ .(.(CONS v1_) v2_)) = .(.(CONS v1_) .(v0_ v2_))",
            ".(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = .(.(v1_ .(v2_ v3_)) .(v0_ v4_))",
            ".(v0_ .(.(v1_ v2_) v3_)) = .(.(v1_ v2_) .(v0_ v3_))",
            ".(v0_ .(v1_ v2_)) = .(v1_ .(v0_ v2_))",
        ];
        for second in ruless[1].iter().map(|rule| rule.display(lex.signature())) {
            assert!(seconds_expected.contains(&second.as_str()));
        }
    }

    #[test]
    fn compute_unique_trss_test() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(".(C .(.(CONS .(v0_ v1_)) NIL)) = NIL; .(C .(.(CONS .(v0_ v1_)) v2_)) = .(.(CONS .(v0_ v1_)) .(C v2_));", &mut lex, true, &[])
            .expect("parsed trs");
        let mut types = trs.collect_types();
        let mut vars = trs.find_all_variablizations(&types);
        let ruless = trs.compute_unique_rules(&mut vars, &mut types);
        let combos = trs.compute_unique_trss(&ruless);
        // See augment_trss_test_2
        for combo in &combos {
            for (m, &n) in combo.iter().enumerate() {
                println!("{}", ruless[m][n].pretty(trs.lex.signature()));
            }
            println!();
        }
        assert_eq!(combos.len(), 163);

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

        let misc_trss = vec![
            ".(C .(.(CONS .(v0_ v1_)) v2_)) = v2_;\n.(C .(.(CONS v0_) v1_)) = .(.(CONS v0_) .(C v1_));",
            ".(C .(.(CONS .(v0_ v1_)) v2_)) = v2_;\n.(C .(.(v0_ .(v1_ v2_)) v3_)) = .(.(v0_ .(v1_ v2_)) .(C v3_));",
            ".(C .(.(CONS .(v0_ v1_)) v2_)) = v2_;\n.(C .(.(v0_ v1_) v2_)) = .(.(v0_ v1_) .(C v2_));",
            ".(C .(.(CONS .(v0_ v1_)) v2_)) = v2_;\n.(C .(v0_ v1_)) = .(v0_ .(C v1_));",
            ".(C .(.(CONS .(v0_ v1_)) v2_)) = v2_;\n.(v0_ .(.(CONS .(v1_ v2_)) v3_)) = .(.(CONS .(v1_ v2_)) .(v0_ v3_));",
            ".(C .(.(CONS .(v0_ v1_)) v2_)) = v2_;\n.(v0_ .(.(CONS v1_) v2_)) = .(.(CONS v1_) .(v0_ v2_));",
            ".(C .(.(CONS .(v0_ v1_)) v2_)) = v2_;\n.(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = .(.(v1_ .(v2_ v3_)) .(v0_ v4_));",
            ".(C .(.(CONS .(v0_ v1_)) v2_)) = v2_;\n.(v0_ .(.(v1_ v2_) v3_)) = .(.(v1_ v2_) .(v0_ v3_));",
            ".(C .(.(CONS .(v0_ v1_)) v2_)) = v2_;\n.(v0_ .(v1_ v2_)) = .(v1_ .(v0_ v2_));",
            ".(C .(.(CONS v0_) v1_)) = v1_;\n.(C .(.(v0_ .(v1_ v2_)) v3_)) = .(.(v0_ .(v1_ v2_)) .(C v3_));",
            ".(C .(.(CONS v0_) v1_)) = v1_;\n.(C .(.(v0_ v1_) v2_)) = .(.(v0_ v1_) .(C v2_));",
            ".(C .(.(CONS v0_) v1_)) = v1_;\n.(C .(v0_ v1_)) = .(v0_ .(C v1_));",
            ".(C .(.(CONS v0_) v1_)) = v1_;\n.(v0_ .(.(CONS .(v1_ v2_)) v3_)) = .(.(CONS .(v1_ v2_)) .(v0_ v3_));",
            ".(C .(.(CONS v0_) v1_)) = v1_;\n.(v0_ .(.(CONS v1_) v2_)) = .(.(CONS v1_) .(v0_ v2_));",
            ".(C .(.(CONS v0_) v1_)) = v1_;\n.(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = .(.(v1_ .(v2_ v3_)) .(v0_ v4_));",
            ".(C .(.(CONS v0_) v1_)) = v1_;\n.(v0_ .(.(v1_ v2_) v3_)) = .(.(v1_ v2_) .(v0_ v3_));",
            ".(C .(.(CONS v0_) v1_)) = v1_;\n.(v0_ .(v1_ v2_)) = .(v1_ .(v0_ v2_));",
            ".(C .(.(v0_ .(v1_ v2_)) v3_)) = v3_;\n.(C .(.(CONS v0_) v1_)) = .(.(CONS v0_) .(C v1_));",
            ".(C .(.(v0_ .(v1_ v2_)) v3_)) = v3_;\n.(C .(.(v0_ v1_) v2_)) = .(.(v0_ v1_) .(C v2_));",
            ".(C .(.(v0_ .(v1_ v2_)) v3_)) = v3_;\n.(C .(v0_ v1_)) = .(v0_ .(C v1_));",
            ".(C .(.(v0_ .(v1_ v2_)) v3_)) = v3_;\n.(v0_ .(.(CONS .(v1_ v2_)) v3_)) = .(.(CONS .(v1_ v2_)) .(v0_ v3_));",
            ".(C .(.(v0_ .(v1_ v2_)) v3_)) = v3_;\n.(v0_ .(.(CONS v1_) v2_)) = .(.(CONS v1_) .(v0_ v2_));",
            ".(C .(.(v0_ .(v1_ v2_)) v3_)) = v3_;\n.(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = .(.(v1_ .(v2_ v3_)) .(v0_ v4_));",
            ".(C .(.(v0_ .(v1_ v2_)) v3_)) = v3_;\n.(v0_ .(.(v1_ v2_) v3_)) = .(.(v1_ v2_) .(v0_ v3_));",
            ".(C .(.(v0_ .(v1_ v2_)) v3_)) = v3_;\n.(v0_ .(v1_ v2_)) = .(v1_ .(v0_ v2_));",
            ".(C .(.(v0_ v1_) v2_)) = v2_;\n.(C .(v0_ v1_)) = .(v0_ .(C v1_));",
            ".(C .(.(v0_ v1_) v2_)) = v2_;\n.(v0_ .(.(CONS .(v1_ v2_)) v3_)) = .(.(CONS .(v1_ v2_)) .(v0_ v3_));",
            ".(C .(.(v0_ v1_) v2_)) = v2_;\n.(v0_ .(.(CONS v1_) v2_)) = .(.(CONS v1_) .(v0_ v2_));",
            ".(C .(.(v0_ v1_) v2_)) = v2_;\n.(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = .(.(v1_ .(v2_ v3_)) .(v0_ v4_));",
            ".(C .(.(v0_ v1_) v2_)) = v2_;\n.(v0_ .(.(v1_ v2_) v3_)) = .(.(v1_ v2_) .(v0_ v3_));",
            ".(C .(.(v0_ v1_) v2_)) = v2_;\n.(v0_ .(v1_ v2_)) = .(v1_ .(v0_ v2_));",
            ".(C .(v0_ v1_)) = v1_;\n.(v0_ .(.(CONS .(v1_ v2_)) v3_)) = .(.(CONS .(v1_ v2_)) .(v0_ v3_));",
            ".(C .(v0_ v1_)) = v1_;\n.(v0_ .(.(CONS v1_) v2_)) = .(.(CONS v1_) .(v0_ v2_));",
            ".(C .(v0_ v1_)) = v1_;\n.(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = .(.(v1_ .(v2_ v3_)) .(v0_ v4_));",
            ".(C .(v0_ v1_)) = v1_;\n.(v0_ .(.(v1_ v2_) v3_)) = .(.(v1_ v2_) .(v0_ v3_));",
            ".(C .(v0_ v1_)) = v1_;\n.(v0_ .(v1_ v2_)) = .(v1_ .(v0_ v2_));",
            ".(v0_ .(.(CONS .(v1_ v2_)) v3_)) = v3_;\n.(C .(.(CONS v0_) v1_)) = .(.(CONS v0_) .(C v1_));",
            ".(v0_ .(.(CONS .(v1_ v2_)) v3_)) = v3_;\n.(C .(.(v0_ .(v1_ v2_)) v3_)) = .(.(v0_ .(v1_ v2_)) .(C v3_));",
            ".(v0_ .(.(CONS .(v1_ v2_)) v3_)) = v3_;\n.(C .(.(v0_ v1_) v2_)) = .(.(v0_ v1_) .(C v2_));",
            ".(v0_ .(.(CONS .(v1_ v2_)) v3_)) = v3_;\n.(C .(v0_ v1_)) = .(v0_ .(C v1_));",
            ".(v0_ .(.(CONS .(v1_ v2_)) v3_)) = v3_;\n.(v0_ .(.(CONS v1_) v2_)) = .(.(CONS v1_) .(v0_ v2_));",
            ".(v0_ .(.(CONS .(v1_ v2_)) v3_)) = v3_;\n.(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = .(.(v1_ .(v2_ v3_)) .(v0_ v4_));",
            ".(v0_ .(.(CONS .(v1_ v2_)) v3_)) = v3_;\n.(v0_ .(.(v1_ v2_) v3_)) = .(.(v1_ v2_) .(v0_ v3_));",
            ".(v0_ .(.(CONS .(v1_ v2_)) v3_)) = v3_;\n.(v0_ .(v1_ v2_)) = .(v1_ .(v0_ v2_));",
            ".(v0_ .(.(CONS v1_) v2_)) = v2_;\n.(C .(.(v0_ .(v1_ v2_)) v3_)) = .(.(v0_ .(v1_ v2_)) .(C v3_));",
            ".(v0_ .(.(CONS v1_) v2_)) = v2_;\n.(C .(.(v0_ v1_) v2_)) = .(.(v0_ v1_) .(C v2_));",
            ".(v0_ .(.(CONS v1_) v2_)) = v2_;\n.(C .(v0_ v1_)) = .(v0_ .(C v1_));",
            ".(v0_ .(.(CONS v1_) v2_)) = v2_;\n.(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = .(.(v1_ .(v2_ v3_)) .(v0_ v4_));",
            ".(v0_ .(.(CONS v1_) v2_)) = v2_;\n.(v0_ .(.(v1_ v2_) v3_)) = .(.(v1_ v2_) .(v0_ v3_));",
            ".(v0_ .(.(CONS v1_) v2_)) = v2_;\n.(v0_ .(v1_ v2_)) = .(v1_ .(v0_ v2_));",
            ".(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = v4_;\n.(C .(.(CONS v0_) v1_)) = .(.(CONS v0_) .(C v1_));",
            ".(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = v4_;\n.(C .(.(v0_ v1_) v2_)) = .(.(v0_ v1_) .(C v2_));",
            ".(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = v4_;\n.(C .(v0_ v1_)) = .(v0_ .(C v1_));",
            ".(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = v4_;\n.(v0_ .(.(CONS v1_) v2_)) = .(.(CONS v1_) .(v0_ v2_));",
            ".(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = v4_;\n.(v0_ .(.(v1_ v2_) v3_)) = .(.(v1_ v2_) .(v0_ v3_));",
            ".(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = v4_;\n.(v0_ .(v1_ v2_)) = .(v1_ .(v0_ v2_));",
            ".(v0_ .(.(v1_ v2_) v3_)) = v3_;\n.(C .(v0_ v1_)) = .(v0_ .(C v1_));",
            ".(v0_ .(.(v1_ v2_) v3_)) = v3_;\n.(v0_ .(v1_ v2_)) = .(v1_ .(v0_ v2_));",
            ".(C v0_) = NIL;\n.(v0_ .(.(CONS .(v1_ v2_)) v3_)) = .(.(CONS .(v1_ v2_)) .(v0_ v3_));",
            ".(C v0_) = NIL;\n.(v0_ .(.(CONS v1_) v2_)) = .(.(CONS v1_) .(v0_ v2_));",
            ".(C v0_) = NIL;\n.(v0_ .(.(v1_ .(v2_ v3_)) v4_)) = .(.(v1_ .(v2_ v3_)) .(v0_ v4_));",
            ".(C v0_) = NIL;\n.(v0_ .(.(v1_ v2_) v3_)) = .(.(v1_ v2_) .(v0_ v3_));",
            ".(C v0_) = NIL;\n.(v0_ .(v1_ v2_)) = .(v1_ .(v0_ v2_));",
        ];

        let trs_strs = combos
            .into_iter()
            .map(|combo| {
                let mut trs = trs.clone();
                for (m, n) in combo.into_iter().enumerate() {
                    trs.utrs.rules[m] = ruless[m][n].clone();
                }
                trs.to_string()
            })
            .collect_vec();

        for rule1 in &rule_1s {
            for rule2 in &rule_2s {
                assert!(trs_strs.contains(&format!("{}\n{}", rule1, rule2)));
            }
        }
        for misc_trs in &misc_trss {
            assert!(trs_strs.contains(&misc_trs.to_string()));
        }
        assert_eq!(trs_strs.len(), 163);
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

        let trss = trs.variablize().unwrap();
        // see augment_trss_test_2
        for trs in trss.iter().sorted_by_key(|trs| trs.size()) {
            println!("{}", trs.to_string().lines().join(" "));
        }
        assert_eq!(trss.len(), 163);
    }
}

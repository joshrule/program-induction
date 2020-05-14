use itertools::Itertools;
use polytype::{Context as TypeContext, Type};
use std::collections::HashMap;
use term_rewriting::{Context, Operator, Place, Rule, RuleContext, Term, Variable};
use trs::{as_result, Environment, Lexicon, SampleError, TRS};
use utils::weighted_permutation;

pub type Recursion = (Term, Place, Place, Type);
type Case = (Option<Rule>, Rule);
type Unroll = (Rule, Vec<Rule>);

impl<'a, 'b> TRS<'a, 'b> {
    pub fn recurse(&self, n_sampled: usize) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        let (trss, ns): (Vec<_>, Vec<_>) = self
            .find_all_recursions()
            .into_iter()
            .filter_map(|recursion| {
                self.try_recursion(&recursion)
                    .ok()
                    .and_then(|solution| self.adopt_recursion(solution))
            })
            .unzip();
        as_result(weighted_permutation(&trss, &ns, Some(n_sampled)))
    }
    pub fn recurse_by(&self, recursion: &Recursion) -> Option<TRS<'a, 'b>> {
        self.try_recursion(recursion)
            .ok()
            .and_then(|solution| self.adopt_recursion(solution))
            .map(|(trs, _)| trs)
    }
    fn adopt_recursion(
        &self,
        (lex, solution): (Lexicon<'b>, Vec<(Rule, Vec<Rule>)>),
    ) -> Option<(TRS<'a, 'b>, f64)> {
        let (old_rules, new_ruless): (Vec<_>, Vec<_>) = solution.into_iter().unzip();
        let mut new_rules = TRS::reorder_rules(new_ruless);
        self.filter_background(&mut new_rules);
        match new_rules.len() {
            1 => None,
            n => {
                let mut trs = TRS::new_unchecked(
                    &lex,
                    self.utrs.is_deterministic(),
                    self.background,
                    self.utrs.rules.clone(),
                );
                trs.remove_clauses(&old_rules).ok()?;
                trs.prepend_clauses(new_rules).ok()?;
                let trs = trs.smart_delete(0, n).ok()?;
                Some((trs, (1.5 as f64).powi(n as i32)))
            }
        }
    }
    fn reorder_rules(new_ruless: Vec<Vec<Rule>>) -> Vec<Rule> {
        let mut new_bases = vec![];
        let mut new_recursives = vec![];
        for mut new_rules in new_ruless {
            // put base case in new bases
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
        new_rules
    }
    pub(crate) fn collect_recursive_fns<'c>(
        map: &HashMap<Place, Type>,
        lex: &Lexicon,
        rule: &'c Rule,
    ) -> Vec<(&'c Term, Place, Type)> {
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
                let mut ctx = lex.0.ctx.clone();
                if tps.len() == 1 && ctx.unify(tp, tps[0]).is_ok() {
                    let new_tp = tp.apply(&ctx);
                    fns.push((rule.at(k).unwrap(), k.to_vec(), new_tp));
                }
            }
        }
        fns
    }
    pub(crate) fn partition_subrules(rule: &Rule) -> (Vec<Place>, Vec<Place>) {
        rule.subterms()
            .into_iter()
            .skip(1)
            .map(|x| x.1)
            .partition(|x| x[0] == 0)
    }
    pub fn find_all_recursions(&self) -> Vec<Recursion> {
        self.utrs
            .clauses()
            .iter()
            .flat_map(|c| self.find_recursions(c))
            .unique()
            .collect_vec()
    }
    /// This function returns a `Vec` of (f, lhs_place, rhs_place, type)
    /// - f: some potentially recursive function (i.e. f: a -> a)
    /// - lhs_place: some LHS subterm which could be f's input (lhs_place: a)
    /// - rhs_place: some RHS subterm which could be f's output (rhs_place: a)
    /// - type: the type of the recursed object
    fn find_recursions(&self, rule: &Rule) -> Vec<Recursion> {
        let mut map = HashMap::new();
        let mut ctx = self.lex.0.ctx.clone();
        let mut env = Environment::from_vars(&rule.variables(), &mut ctx);
        if self
            .lex
            .infer_rule(rule, &mut map, &mut env, &mut ctx)
            .is_err()
        {
            return vec![];
        }
        let fs = TRS::collect_recursive_fns(&map, &self.lex, rule);
        let (lhss, rhss) = TRS::partition_subrules(rule);
        let mut transforms = vec![];
        for (f, place, tp) in fs {
            for lhs_place in &lhss {
                let outer_snapshot = ctx.len();
                let diff_place = place != *lhs_place;
                let trivial = lhs_place.starts_with(&place[0..place.len() - 1])
                    && lhs_place.ends_with(&[1])
                    && lhs_place.len() == place.len();
                if diff_place && !trivial && ctx.unify(&tp, &map[lhs_place]).is_ok() {
                    for rhs_place in &rhss {
                        let inner_snapshot = ctx.len();
                        if ctx.unify(&tp, &map[rhs_place]).is_ok() {
                            let tp = tp.apply(&ctx);
                            let mut context = RuleContext::from(rule);
                            let lhs_term = context.at(lhs_place).unwrap().clone();
                            context = context.replace(&lhs_place, Context::Hole).unwrap();
                            let context_vars = context.lhs.variables();
                            if !lhs_term
                                .variables()
                                .iter()
                                .all(|v| context_vars.contains(v))
                            {
                                continue;
                            }
                            context = context.replace(&rhs_place, Context::Hole).unwrap();
                            if !RuleContext::is_valid(&context.lhs, &context.rhs) {
                                continue;
                            }
                            transforms.push((f.clone(), lhs_place.clone(), rhs_place.clone(), tp));
                        }
                        ctx.rollback(inner_snapshot);
                    }
                }
                ctx.rollback(outer_snapshot);
            }
        }
        transforms
    }
    fn try_recursion(&self, t: &Recursion) -> Result<(Lexicon<'b>, Vec<Unroll>), SampleError> {
        // Collect the full transform for each rule that can be transformed.
        let mut lex = self.lex.clone();
        let op = lex.has_op(Some("."), 2)?;
        let new_ruless = self
            .clauses()
            .into_iter()
            .filter_map(|(_, rule)| {
                TRS::transform_rule(t, op, &rule, &mut lex).map(|rs| (rule, rs))
            })
            .collect_vec();
        // Reconcile the base cases.
        let mut new_ruless =
            TRS::reconcile_base_cases(new_ruless).ok_or(SampleError::OptionsExhausted)?;
        // Put the base cases first.
        new_ruless
            .iter_mut()
            .for_each(|(_, new_rules)| new_rules.reverse());
        as_result(new_ruless).map(|new_ruless| (lex, new_ruless))
    }
    fn transform_rule(
        (f, lhs, rhs, tp): &Recursion,
        op: Operator,
        rule: &Rule,
        lex: &mut Lexicon,
    ) -> Option<Vec<Case>> {
        let mut transforms = vec![];
        let mut basecase = rule.clone();
        while let Some((rec, base)) = TRS::transform_inner(f, lhs, rhs, op, &basecase, lex, tp).ok()
        {
            // Only continue for a novel, nontrivial base case (to avoid loops)
            if base.lhs != base.rhs().unwrap()
                && !transforms
                    .iter()
                    .any(|(_, b)| Rule::alpha(b, &base).is_some())
            {
                transforms.push((Some(rec), basecase));
                basecase = base;
            } else {
                break;
            }
        }
        if transforms.is_empty() {
            None
        } else {
            transforms.push((None, basecase));
            Some(transforms)
        }
    }
    fn reconcile_base_cases(mut ruless: Vec<(Rule, Vec<Case>)>) -> Option<Vec<(Rule, Vec<Rule>)>> {
        loop {
            if ruless.iter().any(|(_, rules)| rules.is_empty()) {
                return None;
            } else if TRS::check_base_cases(&ruless) {
                break;
            } else {
                for (_, rules) in ruless.iter_mut() {
                    rules.pop().as_ref()?;
                }
            }
        }
        let new_ruless = ruless
            .into_iter()
            .map(|(old, rules)| {
                let n = rules.len() - 1;
                let new_rules = rules
                    .into_iter()
                    .enumerate()
                    .map(|(i, (rec, base))| if i < n { rec.unwrap() } else { base })
                    .collect_vec();
                (old, new_rules)
            })
            .collect_vec();
        Some(new_ruless)
    }
    fn check_base_cases(ruless: &[(Rule, Vec<Case>)]) -> bool {
        // Basecases are good if no rules have shared LHSs but differing RHSs
        ruless
            .iter()
            .map(|(_, rules)| rules.last())
            .combinations(2)
            .all(|bs| match (&bs[0], &bs[1]) {
                (Some(x), Some(y)) => Term::alpha(&[(&x.1.lhs, &y.1.lhs)]).is_none() || x.1 == y.1,
                _ => false,
            })
    }
    fn transform_inner(
        f: &Term,
        lhs_place: &[usize],
        rhs_place: &[usize],
        op: Operator,
        rule: &Rule,
        lex: &mut Lexicon,
        var_type: &Type,
    ) -> Result<(Rule, Rule), SampleError> {
        // Perform necessary checks:
        // 0. the rule typechecks;
        let mut ctx = lex.0.ctx.clone();
        let mut map = HashMap::new();
        let mut env = Environment::from_vars(&rule.variables(), &mut ctx);
        lex.infer_rule(rule, &mut map, &mut env, &mut ctx)?;
        // 1. f is at the head of the LHS; and
        TRS::leftmost_symbol_matches(&rule, f, lex, &map)?;
        // 2. lhs_place and rhs_place exist and have the appropriate types.
        let lhs_structure = TRS::check_place(rule, lhs_place, var_type, &mut ctx, &map)?;
        let rhs_structure = TRS::check_place(rule, rhs_place, var_type, &mut ctx, &map)?;
        // 3. Check that we can perform replacement without invalidating rule.
        let mut context = RuleContext::from(rule.clone());
        context.canonicalize(&mut HashMap::new());
        context = context
            .replace(&lhs_place, Context::Hole)
            .ok_or(SampleError::Subterm)?;
        context = context
            .replace(&rhs_place, Context::Hole)
            .ok_or(SampleError::Subterm)?;
        if !RuleContext::is_valid(&context.lhs, &context.rhs) {
            return Err(SampleError::Subterm);
        }
        let id = context.variables().len();
        let new_var = Variable(id);
        // Swap lhs_structure for: new_var.
        let mut rec = rule.clone();
        rec.canonicalize(&mut HashMap::new());
        rec = rec
            .replace(lhs_place, Term::Variable(new_var))
            .ok_or(SampleError::Subterm)?;
        // Swap rhs_structure for: f new_var.
        let new_subterm = Term::Application {
            op,
            args: vec![f.clone(), Term::Variable(new_var)],
        };
        rec = rec
            .replace(rhs_place, new_subterm)
            .ok_or(SampleError::Subterm)?;
        // Create the rule: f lhs_structure = rhs_structure.
        let new_lhs = Term::Application {
            op,
            args: vec![f.clone(), lhs_structure.clone()],
        };
        let base = Rule::new(new_lhs, vec![rhs_structure.clone()]).ok_or(SampleError::Subterm)?;
        // Return.
        Ok((rec, base))
    }
    fn leftmost_symbol_matches(
        rule: &Rule,
        f: &Term,
        lex: &Lexicon,
        map: &HashMap<Place, Type>,
    ) -> Result<(), SampleError> {
        TRS::collect_recursive_fns(map, lex, rule)
            .iter()
            .find(|(term, _, _)| Term::alpha(&[(term, f)]).is_some())
            .map(|_| ())
            .ok_or(SampleError::Subterm)
    }
    fn check_place<'c>(
        rule: &'c Rule,
        place: &[usize],
        tp: &Type,
        ctx: &mut TypeContext,
        map: &HashMap<Place, Type>,
    ) -> Result<&'c Term, SampleError> {
        map.get(place)
            .and_then(|place_tp| ctx.unify(tp, place_tp).ok())
            .and_then(|_| rule.at(place))
            .ok_or(SampleError::Subterm)
    }
}

#[cfg(test)]
mod tests {
    use polytype::Context as TypeContext;
    use std::collections::HashMap;
    use trs::{
        parser::{parse_lexicon, parse_rule, parse_term, parse_trs},
        Environment, Lexicon, TRS,
    };

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
    fn collect_recursive_fns_test() {
        let mut lex = create_test_lexicon();
        let rule = parse_rule(
            "C (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2 ) (CONS (DIGIT 0) NIL)))) = (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2 ) NIL)))",
            &mut lex,
        )
            .expect("parsed rule");
        let mut map = HashMap::new();
        let mut ctx = lex.0.ctx.clone();
        let mut env = Environment::from_vars(&rule.variables(), &mut ctx);
        lex.infer_rule(&rule, &mut map, &mut env, &mut ctx).unwrap();
        let fs = TRS::collect_recursive_fns(&map, &lex, &rule);
        assert_eq!(fs.len(), 1);
        assert_eq!(
            format!(
                "{} {:?} {}",
                fs[0].0.display(lex.signature()),
                fs[0].1,
                fs[0].2
            ),
            "C [0, 0] list"
        );

        let rule = parse_rule("C = C", &mut lex).expect("parsed rule");
        let mut map = HashMap::new();
        let mut ctx = lex.0.ctx.clone();
        let mut env = Environment::from_vars(&rule.variables(), &mut ctx);
        lex.infer_rule(&rule, &mut map, &mut env, &mut ctx).unwrap();
        let fs = TRS::collect_recursive_fns(&map, &lex, &rule);
        assert_eq!(fs.len(), 1);
        assert_eq!(
            format!(
                "{} {:?} {}",
                fs[0].0.display(lex.signature()),
                fs[0].1,
                fs[0].2
            ),
            "C [0] list"
        );
    }
    #[test]
    fn find_transforms_test() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(
            "C (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2 ) (CONS (DIGIT 0) NIL)))) = (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2 ) NIL)));",
            &mut lex,
            true,
            &[]
        )
            .expect("parsed rule");
        let transforms = trs.find_recursions(&trs.utrs.rules[0]);
        for (t, p1, p2, tp) in &transforms {
            println!("{} {:?} {:?} {}", t.pretty(&lex.signature()), p1, p2, tp);
        }
        assert_eq!(16, transforms.len());
    }
    #[test]
    fn find_all_transforms_test() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(
            "C (CONS (DIGIT 6) (CONS (DECC (DIGIT 6) 3) (CONS (DECC (DIGIT 8) 6 ) (CONS (DIGIT 8) (CONS (DIGIT 9) (CONS (DIGIT 4) (CONS (DIGIT 7) NIL))))))) = (CONS (DIGIT 6) (CONS (DIGIT 6) (CONS (DIGIT 6) (CONS (DIGIT 6) (CONS (DECC (DIGIT 6) 3) (CONS (DECC (DIGIT 8) 6 ) (CONS (DIGIT 8) (CONS (DIGIT 9) (CONS (DIGIT 4) (CONS (DIGIT 7) NIL))))))))));",
            &mut lex,
            true,
            &[]
        )
            .expect("parsed trs");
        let transforms = trs.find_all_recursions();
        for (t, p1, p2, tp) in &transforms {
            println!("{} {:?} {:?} {}", t.pretty(&lex.signature()), p1, p2, tp);
        }
        assert_eq!(77, transforms.len());
    }
    #[test]
    fn transform_inner_test() {
        let mut lex = create_test_lexicon();
        let op = lex.has_op(Some("."), 2).unwrap();
        let rule = parse_rule(
            "C (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2 ) (CONS (DIGIT 0) NIL)))) = (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2 ) NIL)))",
            &mut lex,
        )
            .expect("parsed rule");
        let f = parse_term("C", &mut lex).expect("parsed term");
        let lhs_place = vec![0, 1, 1];
        let rhs_place = vec![1, 1];
        let mut map = HashMap::new();
        let mut ctx = lex.0.ctx.clone();
        let mut env = Environment::from_vars(&rule.variables(), &mut ctx);
        lex.infer_rule(&rule, &mut map, &mut env, &mut ctx).ok();
        let result = TRS::transform_inner(
            &f,
            &lhs_place,
            &rhs_place,
            op,
            &rule,
            &mut lex,
            &map[&lhs_place],
        );
        assert!(result.is_ok());
        let (new_rule1, new_rule2) = result.unwrap();
        let sig = &lex.0.signature;

        assert_eq!(
            "C (CONS (DIGIT 9) v0_) = CONS (DIGIT 9) (C v0_)",
            new_rule1.pretty(sig),
        );
        assert_eq!(
            "C (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2) (CONS (DIGIT 0) []))) = CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2) [])",
            new_rule2.pretty(sig),
        );
    }
    #[test]
    fn transform_test() {
        let trs_str = ".(C .(.(CONS .(DIGIT 2)) .(.(CONS .(DIGIT 3)) .(.(CONS .(.(DECC .(DIGIT 1)) 0)) .(.(CONS .(DIGIT 9)) .(.(CONS .(.(DECC .(DIGIT 2)) 0)) .(.(CONS .(DIGIT 3)) .(.(CONS .(.(DECC .(DIGIT 7)) 7)) .(.(CONS .(DIGIT 0)) .(.(CONS .(.(DECC .(DIGIT 5)) 4)) NIL)))))))))) = .(.(CONS .(DIGIT 2)) .(.(CONS .(DIGIT 3)) .(.(CONS .(.(DECC .(DIGIT 1)) 0)) .(.(CONS .(DIGIT 9)) .(.(CONS .(.(DECC .(DIGIT 2)) 0)) .(.(CONS .(DIGIT 3)) .(.(CONS .(.(DECC .(DIGIT 7)) 7)) .(.(CONS .(DIGIT 0)) NIL))))))));";
        let mut lex = create_test_lexicon();
        let trs = parse_trs(trs_str, &mut lex, true, &[]).expect("parsed trs");
        let f = parse_term("C", &mut lex).expect("parsed term");
        let lhs_place = vec![0, 1, 1];
        let rhs_place = vec![1, 1];
        let tp = tp![list];
        let tuple = (f, lhs_place, rhs_place, tp);

        let result = trs.try_recursion(&tuple);
        assert!(result.is_ok());

        let (lex, mut new_ruless) = result.unwrap();
        assert_eq!(1, new_ruless.len());

        let sig = &lex.0.signature;
        let (old_rule, new_rules) = new_ruless.pop().unwrap();
        println!("{}", old_rule.pretty(sig));
        assert_eq!(trs_str, format!("{};", old_rule.display(sig)));

        for (i, rule) in new_rules.iter().enumerate() {
            println!("{}. {}", i, rule.pretty(sig));
        }
        assert_eq!(9, new_rules.len());
        assert_eq!(
            "C (CONS (DECC (DIGIT 5) 4) []) = []",
            new_rules[0].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DIGIT 0) v0_) = CONS (DIGIT 0) (C v0_)",
            new_rules[1].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DECC (DIGIT 7) 7) v0_) = CONS (DECC (DIGIT 7) 7) (C v0_)",
            new_rules[2].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DIGIT 3) v0_) = CONS (DIGIT 3) (C v0_)",
            new_rules[3].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DECC (DIGIT 2) 0) v0_) = CONS (DECC (DIGIT 2) 0) (C v0_)",
            new_rules[4].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DIGIT 9) v0_) = CONS (DIGIT 9) (C v0_)",
            new_rules[5].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DECC (DIGIT 1) 0) v0_) = CONS (DECC (DIGIT 1) 0) (C v0_)",
            new_rules[6].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DIGIT 3) v0_) = CONS (DIGIT 3) (C v0_)",
            new_rules[7].pretty(sig),
        );
        assert_eq!(
            "C (CONS (DIGIT 2) v0_) = CONS (DIGIT 2) (C v0_)",
            new_rules[8].pretty(sig),
        );
    }
    #[test]
    fn recurse_test_1() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(
            "C (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) (CONS (DECC (DIGIT 5) 4) NIL))))))))) = (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) NIL))))))));",
            &mut lex,
            true,
            &[]
        )
            .expect("parsed TRS");
        let result = trs.recurse(20);
        assert!(result.is_ok());
        let trss = result.unwrap();
        for trs in &trss {
            println!("##\n{}\n##", trs);
        }
        assert_eq!(20, trss.len());
    }
    #[test]
    fn recurse_test_2() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(
            "C (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) (CONS (DECC (DIGIT 5) 4) NIL))))))))) = (CONS (DECC (DIGIT 5) 4)  NIL);C (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2) (CONS (DIGIT 0) NIL)))) = (CONS (DIGIT 0) NIL);",
            &mut lex,
            true,
            &[],
        )
            .expect("parsed TRS");
        let result = trs.recurse(20);
        assert!(result.is_ok());
        let trss = result.unwrap();
        for trs in &trss {
            println!("\n{}\n", trs);
        }
        assert_eq!(18, trss.len());
    }
    #[test]
    fn recurse_test_3() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(
            "C (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) (CONS (DECC (DIGIT 5) 4) NIL))))))))) = (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) NIL))))))));C (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2) (CONS (DIGIT 0) NIL)))) = (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 3) 2 ) NIL)));",
            &mut lex,
            true,
            &[],
        )
            .expect("parsed TRS");
        let result = trs.recurse(20);
        assert!(result.is_ok());
        let trss = result.unwrap();

        for trs in &trss {
            println!("\n{}\n", trs);
        }

        assert_eq!(20, trss.len());
    }
    #[test]
    fn recurse_test_4() {
        let mut lex = parse_lexicon(
            &[
                "C/0: list -> list;",
                "CONS/0: nat -> list -> list;",
                "NIL/0: list;",
                "HEAD/0: list -> nat;",
                "TAIL/0: list -> list;",
                "NIL/0: list -> bool;",
                "EQUAL/0: t1. t1 -> t1 -> bool;",
                "IF/0: t1. bool -> t1 -> t1 -> t1;",
                ">/0: nat -> nat -> bool;",
                "+/0: nat -> nat -> nat;",
                "-/0: nat -> nat -> nat;",
                "TRUE/0: bool;",
                "FALSE/0: bool;",
                "DIGIT/0: int -> nat;",
                "DECC/0: nat -> int -> nat;",
                "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
                "NAN/0: nat;",
                "0/0: int; 1/0: int; 2/0: int;",
                "3/0: int; 4/0: int; 5/0: int;",
                "6/0: int; 7/0: int; 8/0: int;",
                "9/0: int;",
            ]
            .join(" "),
            TypeContext::default(),
        )
        .expect("parsed lexicon");
        let trs = parse_trs("C (v0_ (DIGIT 3) (CONS (DIGIT 0) (CONS (DIGIT 4) NIL))) = CONS (DIGIT 2) (CONS (DIGIT 3) NIL);", &mut lex, true, &[]).expect("parsed TRS");
        println!("{}", trs);
        println!("{}", trs.find_all_recursions().len());
        let result = trs.recurse(20);
        assert!(result.is_ok());
        let trss = result.unwrap();

        for trs in &trss {
            println!("\n{}\n", trs);
        }

        assert_eq!(9, trss.len());
    }
    #[test]
    fn recurse_test_5() {
        let mut lex = parse_lexicon(
            &[
                "C/0: list -> list;",
                "CONS/0: nat -> list -> list;",
                "NIL/0: list;",
                "HEAD/0: list -> nat;",
                "TAIL/0: list -> list;",
                "NIL/0: list -> bool;",
                "EQUAL/0: t1. t1 -> t1 -> bool;",
                "IF/0: t1. bool -> t1 -> t1 -> t1;",
                ">/0: nat -> nat -> bool;",
                "+/0: nat -> nat -> nat;",
                "-/0: nat -> nat -> nat;",
                "TRUE/0: bool;",
                "FALSE/0: bool;",
                "DIGIT/0: int -> nat;",
                "DECC/0: nat -> int -> nat;",
                "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
                "NAN/0: nat;",
                "0/0: int; 1/0: int; 2/0: int;",
                "3/0: int; 4/0: int; 5/0: int;",
                "6/0: int; 7/0: int; 8/0: int;",
                "9/0: int;",
            ]
            .join(" "),
            TypeContext::default(),
        )
        .expect("parsed lexicon");
        let trs = parse_trs(
            "+ (HEAD (v0_ IF)) = IF TRUE - - (HEAD NIL);",
            &mut lex,
            true,
            &[],
        )
        .expect("parsed TRS");
        assert!(trs.recurse(20).is_err());
    }
}

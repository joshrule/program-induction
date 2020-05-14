use itertools::Itertools;
use polytype::atype::{Ty, Type};
use std::collections::HashMap;
use term_rewriting::{Operator, Place, Rule, Term, Variable};
use trs::{rewrite::FactoredSolution, Lexicon, SampleError, TRS};

pub type Composition<'ctx> = (Term, Place, Place, Ty<'ctx>);

impl<'ctx, 'b> TRS<'ctx, 'b> {
    pub fn find_all_compositions(&self) -> Vec<Composition<'ctx>> {
        self.clauses()
            .into_iter()
            .map(|(_, c)| c)
            .flat_map(|c| self.find_compositions(&c))
            .unique()
            .collect_vec()
    }
    fn find_compositions(&self, rule: &Rule) -> Vec<Composition<'ctx>> {
        // Typecheck the rule.
        if let Ok(mut env) = self.lex.infer_rule(rule) {
            let map: HashMap<_, _> = rule
                .subterms()
                .into_iter()
                .zip(&env.tps)
                .map(|((_, p), tp)| (p, *tp))
                .collect();
            // Find the symbols in the rule that might decompose.
            let ss = env.snapshot();
            let fs = TRS::collect_recursive_fns(&map, rule, &mut env.sub);
            env.rollback(ss);
            // For each decomposable symbol:
            let (_, rhss) = TRS::partition_subrules(rule);
            let mut transforms = vec![];
            for (f, place, tp) in fs {
                // Find the argument to the symbol.
                let mut lhs_place = place.to_vec();
                if let Some(position) = lhs_place.last_mut() {
                    *position = 1;
                }
                // If the argument has the appropriate type:
                let outer_snapshot = env.snapshot();
                if Type::unify_with_sub(&[(&tp, &map[&lhs_place])], &mut env.sub).is_ok() {
                    // For each subterm in the RHS:
                    for rhs_place in &rhss {
                        // If the argument has the appropriate type:
                        let inner_snapshot = env.snapshot();
                        if rhs_place.len() > 1
                            && Type::unify_with_sub(&[(&tp, &map[rhs_place])], &mut env.sub).is_ok()
                        {
                            // that item can be decomposed.
                            let tp = tp.apply(&env.sub);
                            transforms.push((f.clone(), lhs_place.clone(), rhs_place.clone(), tp));
                        }
                        env.rollback(inner_snapshot);
                    }
                }
                env.rollback(outer_snapshot);
            }
            transforms
        } else {
            vec![]
        }
    }
    pub fn compose_by(&self, composition: &Composition<'ctx>) -> Option<Self> {
        let clauses = self.clauses().into_iter().map(|(_, c)| c).collect_vec();
        self.try_composition(composition, &clauses)
            .ok()
            .and_then(|solution| self.adopt_composition(solution))
    }
    fn try_composition(
        &self,
        t: &Composition<'ctx>,
        rules: &[Rule],
    ) -> Result<FactoredSolution<'ctx, 'b>, SampleError<'ctx>> {
        let mut lex = self.lex.clone();
        // Identify atoms, including two new operators F and G.
        let op = lex.has_operator(Some("."), 2)?;
        let tp = lex.lex.ctx.arrow(t.3, t.3);
        let mut headmost = t.0.clone();
        headmost.canonicalize(&mut HashMap::new());
        let f = lex.invent_operator(None, 0, &tp);
        let g = lex.invent_operator(None, 0, &tp);
        let x = Variable(headmost.variables().len());
        // Add the rule T x = F (G x).
        let master = vec![TRS::make_txfgx_rule(headmost, f, g, x, op, &lex)?];
        // Process the existing rules into subproblems.
        let mut old_rules = vec![];
        let mut f_subproblem = vec![];
        let mut g_subproblem = vec![];
        for rule in rules {
            if Term::pmatch(&[(&master[0].lhs, &rule.lhs)]).is_some() {
                let t_1 = rule.at(&t.1).ok_or(SampleError::Subterm)?.clone();
                let t_2 = rule.at(&t.2).ok_or(SampleError::Subterm)?.clone();
                let t_3 = rule.rhs().ok_or(SampleError::Subterm)?;
                // Add the rule G x = y', where y' is a subterm of y.
                g_subproblem.push(TRS::make_fxy_rule(g, t_1, t_2.clone(), op, &lex)?);
                // Add the rule F y' = y.
                f_subproblem.push(TRS::make_fxy_rule(f, t_2, t_3, op, &lex)?);
                old_rules.push(rule.clone());
            }
        }
        // Ok((master, new_rules, lex))
        Ok((lex, old_rules, master, f_subproblem, g_subproblem))
    }
    fn make_fxy_rule(
        f: Operator,
        x: Term,
        y: Term,
        op: Operator,
        lex: &Lexicon,
    ) -> Result<Rule, SampleError<'ctx>> {
        let lhs = Term::apply(
            op,
            vec![
                Term::apply(f, vec![], lex.signature()).ok_or(SampleError::Subterm)?,
                x,
            ],
            lex.signature(),
        )
        .ok_or(SampleError::Subterm)?;
        let rhs = y;
        Rule::new(lhs, vec![rhs]).ok_or(SampleError::Subterm)
    }
    fn make_txfgx_rule(
        t: Term,
        f: Operator,
        g: Operator,
        v: Variable,
        op: Operator,
        lex: &Lexicon,
    ) -> Result<Rule, SampleError<'ctx>> {
        let lhs = Term::apply(op, vec![t, Term::Variable(v)], lex.signature())
            .ok_or(SampleError::Subterm)?;
        let rhs = Term::apply(
            op,
            vec![
                Term::apply(f, vec![], lex.signature()).ok_or(SampleError::Subterm)?,
                Term::apply(
                    op,
                    vec![
                        Term::apply(g, vec![], lex.signature()).ok_or(SampleError::Subterm)?,
                        Term::Variable(v),
                    ],
                    lex.signature(),
                )
                .ok_or(SampleError::Subterm)?,
            ],
            lex.signature(),
        )
        .ok_or(SampleError::Subterm)?;
        Rule::new(lhs, vec![rhs]).ok_or(SampleError::Subterm)
    }
    fn adopt_composition(
        &self,
        (lex, old_rules, master, f_sub, g_sub): FactoredSolution<'ctx, 'b>,
    ) -> Option<Self> {
        // Combine rules
        let mut new_rules = vec![];
        for rule in f_sub.into_iter().chain(g_sub).chain(master) {
            if new_rules.iter().all(|r| Rule::alpha(r, &rule).is_none()) {
                new_rules.push(rule);
            }
        }
        self.filter_background(&mut new_rules);
        match new_rules.len() {
            0 | 1 => None,
            n => {
                let mut trs = self.clone();
                trs.lex = lex;
                trs.remove_clauses(&old_rules).ok()?;
                trs.prepend_clauses(new_rules).ok()?;
                let trs = trs.smart_delete(0, n).ok()?;
                Some(trs)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use polytype::atype::{with_ctx, TypeContext};
    use trs::{
        parser::{parse_lexicon, parse_rule, parse_trs},
        Lexicon, TRS,
    };

    fn create_test_lexicon<'ctx, 'b>(ctx: &TypeContext<'ctx>) -> Lexicon<'ctx, 'b> {
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
    fn find_all_compositions_test() {
        with_ctx(1024, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
            let trs = parse_trs(
                "C (CONS (DIGIT 1) (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DIGIT 4) NIL)))) = (CONS (DIGIT 0) (CONS (DIGIT 1) (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DIGIT 4) (CONS (DIGIT 5) NIL))))));",
                &mut lex,
                true,
                &[],
            ).expect("trs");
            let compositions = trs.find_all_compositions();

            let sig = lex.signature();
            for (i, (t, p1, p2, tp)) in compositions.iter().enumerate() {
                println!("{}. {} {:?} {:?} {}", i, t.pretty(&sig), p1, p2, tp);
            }

            assert_eq!(6, compositions.len());
        })
    }
    #[test]
    fn compose_by_test() {
        with_ctx(1024, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
            let trs = parse_trs(
                "C (CONS (DIGIT 1) (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DIGIT 4) NIL)))) = (CONS (DIGIT 0) (CONS (DIGIT 1) (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DIGIT 4) (CONS (DIGIT 5) NIL))))));",
                &mut lex,
                true,
                &[],
            )
                .expect("parsed rule");

            let trss: Vec<_> = trs
                .find_all_compositions()
                .into_iter()
                .map(|c| trs.compose_by(&c))
                .collect::<Option<Vec<_>>>()
                .expect("trss");

            for (i, trs) in trss.iter().enumerate() {
                println!("{}.\n{}\n", i, trs);
            }
            assert_eq!(6, trss.len());
        })
    }
}

use itertools::Itertools;
use polytype::Type;
use std::collections::HashMap;
use term_rewriting::{Operator, Place, Rule, Term, Variable};
use trs::{as_result, rewrite::FactoredSolution, SampleError, TRS};

pub type Composition = (Term, Place, Place, Type);

impl<'a, 'b> TRS<'a, 'b> {
    pub fn compose(&self) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        let (_, clauses): (Vec<usize>, Vec<Rule>) = self.clauses().into_iter().unzip();
        let trss = self
            .find_all_compositions()
            .into_iter()
            .filter_map(|composition| self.try_composition(&composition, &clauses).ok())
            .filter_map(|solution| self.adopt_composition(solution))
            .collect_vec();
        as_result(trss)
    }
    pub fn find_all_compositions(&self) -> Vec<Composition> {
        self.clauses()
            .into_iter()
            .map(|(_, c)| c)
            .flat_map(|c| self.find_compositions(&c))
            .unique()
            .collect_vec()
    }
    pub fn compose_by(&self, composition: &Composition) -> Option<TRS<'a, 'b>> {
        let clauses = self.clauses().into_iter().map(|(_, c)| c).collect_vec();
        self.try_composition(composition, &clauses)
            .ok()
            .and_then(|solution| self.adopt_composition(solution))
    }
    fn find_compositions(&self, rule: &Rule) -> Vec<Composition> {
        // Typecheck the rule.
        let mut ctx = self.lex.0.ctx.clone();
        let mut map = HashMap::new();
        if self.lex.infer_rule(rule, &mut map, &mut ctx).is_err() {
            return vec![];
        }
        // Find the symbols in the rule that might decompose.
        let fs = TRS::collect_recursive_fns(&map, &self.lex, rule);
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
            let outer_snapshot = ctx.len();
            if ctx.unify(&tp, &map[&lhs_place]).is_ok() {
                // For each subterm in the RHS:
                for rhs_place in &rhss {
                    // If the argument has the appropriate type:
                    let inner_snapshot = ctx.len();
                    if rhs_place.len() > 1 && ctx.unify(&tp, &map[rhs_place]).is_ok() {
                        // that item can be decomposed.
                        let tp = tp.apply(&ctx);
                        transforms.push((f.clone(), lhs_place.clone(), rhs_place.clone(), tp));
                    }
                    ctx.rollback(inner_snapshot);
                }
            }
            ctx.rollback(outer_snapshot);
        }
        transforms
    }
    fn try_composition(
        &self,
        t: &Composition,
        rules: &[Rule],
    ) -> Result<FactoredSolution<'b>, SampleError> {
        let mut lex = self.lex.clone();
        // Identify atoms, including two new operators F and G.
        let op = lex.has_op(Some("."), 2)?;
        let tp = Type::arrow(t.3.clone(), t.3.clone());
        let f = lex.invent_operator(None, 0, &tp);
        let g = lex.invent_operator(None, 0, &tp);
        let x = Variable { id: 0 };
        // Add the rule T x = F (G x).
        let master = vec![TRS::make_txfgx_rule(t.0.clone(), f, g, x, op)?];
        // Process the existing rules into subproblems.
        let mut old_rules = vec![];
        let mut f_subproblem = vec![];
        let mut g_subproblem = vec![];
        for rule in rules {
            if Term::pmatch(vec![(&master[0].lhs, &rule.lhs)]).is_some() {
                let t_1 = rule.at(&t.1).ok_or(SampleError::Subterm)?.clone();
                let t_2 = rule.at(&t.2).ok_or(SampleError::Subterm)?.clone();
                let t_3 = rule.rhs().ok_or(SampleError::Subterm)?;
                // Add the rule G x = y', where y' is a subterm of y.
                g_subproblem.push(TRS::make_fxy_rule(g, t_1, t_2.clone(), op)?);
                // Add the rule F y' = y.
                f_subproblem.push(TRS::make_fxy_rule(f, t_2, t_3, op)?);
                old_rules.push(rule.clone());
            }
        }
        // Ok((master, new_rules, lex))
        Ok((lex, old_rules, master, f_subproblem, g_subproblem))
    }
    fn make_fxy_rule(f: Operator, x: Term, y: Term, op: Operator) -> Result<Rule, SampleError> {
        let lhs = Term::apply(
            op,
            vec![Term::apply(f, vec![]).ok_or(SampleError::Subterm)?, x],
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
    ) -> Result<Rule, SampleError> {
        let lhs = Term::apply(op, vec![t, Term::Variable(v)]).ok_or(SampleError::Subterm)?;
        let rhs = Term::apply(
            op,
            vec![
                Term::apply(f, vec![]).ok_or(SampleError::Subterm)?,
                Term::apply(
                    op,
                    vec![
                        Term::apply(g, vec![]).ok_or(SampleError::Subterm)?,
                        Term::Variable(v),
                    ],
                )
                .ok_or(SampleError::Subterm)?,
            ],
        )
        .ok_or(SampleError::Subterm)?;
        Rule::new(lhs, vec![rhs]).ok_or(SampleError::Subterm)
    }
    fn adopt_composition(
        &self,
        (lex, old_rules, master, f_sub, g_sub): FactoredSolution<'b>,
    ) -> Option<TRS<'a, 'b>> {
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
    use polytype::Context as TypeContext;
    use trs::{
        parser::{parse_lexicon, parse_rule, parse_trs},
        Lexicon, TRS,
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
    fn find_compositions_test() {
        let mut lex = create_test_lexicon();
        let trs = TRS::new_unchecked(&lex, true, &[], vec![]);
        let rule = parse_rule(
            "C (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) (CONS (DECC (DIGIT 5) 4) NIL))))))))) = (CONS (DIGIT 9) (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) NIL)))))))))",
            &mut lex,
        )
            .expect("parsed rule");
        let compositions = trs.find_compositions(&rule);

        let sig = lex.signature();
        for (i, (t, p1, p2, tp)) in compositions.iter().enumerate() {
            println!("{}. {} {:?} {:?} {}", i, t.pretty(&sig), p1, p2, tp);
        }

        assert_eq!(9, compositions.len());
    }
    #[test]
    fn compose_test() {
        let mut lex = create_test_lexicon();
        let trs = parse_trs(
            "C (CONS (DIGIT 1) (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DIGIT 4) NIL)))) = (CONS (DIGIT 0) (CONS (DIGIT 1) (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DIGIT 4) (CONS (DIGIT 5) NIL))))));",
            &mut lex,
            true,
            &[],
        )
            .expect("parsed rule");

        let result = trs.compose();
        assert!(result.is_ok());

        let trss = result.unwrap();
        for (i, trs) in trss.iter().enumerate() {
            println!("{}.\n{}\n", i, trs);
        }
        assert_eq!(6, trss.len());
    }
}

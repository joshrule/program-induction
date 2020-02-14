use super::{super::as_result, Lexicon, SampleError, TRS};
use itertools::Itertools;
use polytype::Type;
use std::collections::HashMap;
use term_rewriting::{Operator, Place, Rule, Term, Variable};

type Transform = (Term, Vec<usize>, Vec<usize>, Type);
type Case<'a> = (&'a Rule, Vec<Rule>);

impl<'a, 'b> TRS<'a, 'b> {
    pub fn compose(&self) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        let clauses = self.clauses().into_iter().map(|(_, x)| x).collect_vec();
        let snapshot = self.lex.snapshot();
        let op = self.lex.has_op(Some("."), 2)?;
        let trss = clauses
            .iter()
            .flat_map(|r| TRS::find_compositions(r, &self.lex))
            .unique()
            .filter_map(|c| {
                TRS::try_composition(&c, op, &clauses, self.lex.clone())
                    .ok()
                    .and_then(|(master, new_ruless, lex)| {
                        self.adopt_composition(master, new_ruless, lex)
                    })
            })
            .collect_vec();
        self.lex.rollback(snapshot);
        as_result(trss)
    }
    fn find_compositions(rule: &Rule, lex: &Lexicon) -> Vec<(Term, Place, Place, Type)> {
        let mut map = HashMap::new();
        if lex.infer_rule(rule, &mut map).keep().is_err() {
            return vec![];
        }
        let fs = TRS::collect_recursive_fns(&map, lex, rule);
        let (_, rhss) = TRS::partition_subrules(rule);
        let mut transforms = vec![];
        for (f, place, tp) in fs {
            let mut lhs_place = place.to_vec();
            if let Some(position) = lhs_place.last_mut() {
                *position = 1;
            }
            let outer_snapshot = lex.snapshot();
            if lex.unify(&tp, &map[&lhs_place]).is_ok() {
                for rhs_place in &rhss {
                    let inner_snapshot = lex.snapshot();
                    if rhs_place.len() > 1 && lex.unify(&tp, &map[rhs_place]).is_ok() {
                        let tp = tp.apply(&lex.0.ctx.read().expect("poisoned context"));
                        transforms.push((f.clone(), lhs_place.clone(), rhs_place.clone(), tp));
                    }
                    lex.rollback(inner_snapshot);
                }
            }
            lex.rollback(outer_snapshot);
        }
        transforms
    }
    fn try_composition<'c>(
        t: &Transform,
        op: Operator,
        rules: &'c [Rule],
        mut lex: Lexicon<'b>,
    ) -> Result<(Rule, Vec<Case<'c>>, Lexicon<'b>), SampleError> {
        // 1. Define two new operators F and G.
        let tp = Type::arrow(t.3.clone(), t.3.clone());
        let f = lex.invent_operator(None, 0, &tp);
        let g = lex.invent_operator(None, 0, &tp);
        let v = lex.invent_variable(&t.3);
        // 2. Add the rule T x = F (G x).
        let master = TRS::make_txfgx_rule(t.0.clone(), f, g, v, op)?;
        let mut new_rules = vec![];
        for rule in rules {
            if Term::pmatch(vec![(&master.lhs, &rule.lhs)]).is_some() {
                let t_1 = rule.at(&t.1).ok_or(SampleError::Subterm)?.clone();
                let t_2 = rule.at(&t.2).ok_or(SampleError::Subterm)?.clone();
                let t_3 = rule.rhs().ok_or(SampleError::Subterm)?;
                // 1. Add the rule G x = y', where y' is a subterm of y
                let g_rule = TRS::make_fxy_rule(g, t_1, t_2.clone(), op)?;
                // 2. Add the rule F y' = y
                let f_rule = TRS::make_fxy_rule(f, t_2, t_3, op)?;
                new_rules.push((rule, vec![g_rule, f_rule]));
            }
        }
        Ok((master, new_rules, lex))
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
        master: Rule,
        solution: Vec<Case>,
        lex: Lexicon<'b>,
    ) -> Option<TRS<'a, 'b>> {
        let (old_rules, new_ruless): (Vec<_>, Vec<_>) = solution.into_iter().unzip();
        let mut new_rules = vec![];
        for rules in new_ruless {
            for rule in rules {
                if new_rules.iter().all(|r| Rule::alpha(r, &rule).is_none()) {
                    new_rules.push(rule);
                }
            }
        }
        self.filter_background(&mut new_rules);
        let old_rules = old_rules.into_iter().cloned().collect_vec(); // HACK
        match new_rules.len() {
            1 => None,
            n => {
                let mut trs = self.clone();
                trs.lex = lex;
                trs.remove_clauses(&old_rules).ok()?;
                trs.utrs.push(master).ok()?;
                trs.prepend_clauses(new_rules).ok()?;
                let trs = trs.smart_delete(0, n).ok()?;
                Some(trs)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Lexicon, TRS};
    use polytype::Context as TypeContext;
    use trs::parser::{parse_lexicon, parse_rule, parse_trs};

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
        let rule = parse_rule(
            "C (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) (CONS (DECC (DIGIT 5) 4) NIL))))))))) = (CONS (DIGIT 9) (CONS (DIGIT 2) (CONS (DIGIT 3) (CONS (DECC (DIGIT 1) 0 ) (CONS (DIGIT 9) (CONS (DECC (DIGIT 2) 0) (CONS (DIGIT 3) (CONS (DECC (DIGIT 7) 7) (CONS (DIGIT 0) NIL)))))))))",
            &mut lex,
        )
            .expect("parsed rule");
        let compositions = TRS::find_compositions(&rule, &lex);

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

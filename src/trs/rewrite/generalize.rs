use super::{Lexicon, SampleError, TRS};
use itertools::Itertools;
use polytype::Type;
use std::collections::HashMap;
use term_rewriting::{Atom, Context, Operator, Rule, Term, Variable};

impl<'a, 'b> TRS<'a, 'b> {
    pub fn generalize(&self, data: &[Rule]) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        let all_rules = self.clauses_for_learning(data)?;
        let mut trs = self.clone();
        let (lhs_context, clauses) = TRS::find_lhs_context(&all_rules)?;
        let (rhs_context, clauses) = TRS::find_rhs_context(&clauses)?;
        let mut new_rules =
            TRS::generalize_clauses(&trs.lex, &lhs_context, &rhs_context, &clauses)?;
        trs.remove_clauses(&clauses)?;
        let start = trs.len();
        trs.filter_background(&mut new_rules);
        trs.append_clauses(new_rules)?;
        let stop = trs.len();
        Ok(vec![trs.smart_delete(start, stop)?])
    }
    fn find_lhs_context(clauses: &[Rule]) -> Result<(Context, Vec<Rule>), SampleError> {
        TRS::find_shared_context(clauses, |c| c.lhs.clone(), 1)
    }
    fn find_rhs_context(clauses: &[Rule]) -> Result<(Context, Vec<Rule>), SampleError> {
        TRS::find_shared_context(clauses, |c| c.rhs().unwrap(), 3) // constant is a HACK
    }
    fn find_shared_context<T>(
        clauses: &[Rule],
        f: T,
        max_holes: usize,
    ) -> Result<(Context, Vec<Rule>), SampleError>
    where
        T: Fn(&Rule) -> Term,
    {
        // Pair each clause with its term and sort by size.
        let mut fs = clauses
            .iter()
            .map(|c| {
                let t = f(c);
                (t.size(), t, c)
            })
            .collect_vec();
        fs.sort_by_key(|x| x.0);

        // Find the largest contexts that appear in 2 or more terms
        let mut lscs = vec![Context::Hole];
        let mut lsc_size = lscs[0].size();
        while let Some((ref_size, ref_term, _)) = fs.pop() {
            // if t1 is smaller than lsc, you're done
            if ref_size <= lsc_size {
                break;
            }
            for (size, term, _) in fs.iter().rev() {
                // If t2 is smaller than lsc, break (could drop t2 and remaining items).
                if *size <= lsc_size {
                    break;
                }
                // Find c, the largest shared context between t1 and t2.
                let c = TRS::largest_shared_context(&ref_term, term, max_holes)
                    .ok_or(SampleError::OptionsExhausted)?;
                let c_size = c.size();
                // Conditionally update lsc and sharers.
                if c_size > lsc_size {
                    lscs = vec![c];
                    lsc_size = lscs[0].size();
                } else if c_size == lsc_size && !lscs.contains(&c) {
                    lscs.push(c);
                }
            }
        }

        // Now, find the context which shares the most clauses.
        lscs.into_iter()
            .map(|lsc| {
                let sharers = clauses
                    .iter()
                    .filter(|c| Context::generalize(vec![(&lsc, &Context::from(f(c)))]).is_some())
                    .cloned()
                    .collect_vec();
                (lsc, sharers)
            })
            .max_by_key(|(_, sharers)| sharers.len())
            .ok_or(SampleError::OptionsExhausted)
    }
    fn largest_shared_context(t1: &Term, t2: &Term, max_holes: usize) -> Option<Context> {
        // TODO: This fn craps out if max_holes == 0.
        let mut context = TRS::lsc_helper(t1, t2, &mut HashMap::new());
        let mut holes = context.holes();
        if max_holes == 0 && !holes.is_empty() {
            return None;
        }
        while holes.len() > max_holes {
            holes.sort_by_key(|p| p.len());
            let deepest = holes.pop().unwrap();
            if deepest.is_empty() {
                return None;
            }
            context = context.replace(&deepest[..(deepest.len() - 1)], Context::Hole)?;
            holes = context.holes();
        }
        Some(context)
    }
    fn lsc_helper(t1: &Term, t2: &Term, map: &mut HashMap<Variable, Variable>) -> Context {
        match (t1, t2) {
            (Term::Variable(v1), Term::Variable(v2)) => {
                let v = map.entry(*v1).or_insert(*v2);
                if v == v2 {
                    Context::Variable(*v1)
                } else {
                    Context::Hole
                }
            }
            (
                Term::Application {
                    op: op1,
                    args: ref args1,
                },
                Term::Application {
                    op: op2,
                    args: ref args2,
                },
            ) => {
                if op1 != op2 {
                    Context::Hole
                } else {
                    Context::Application {
                        op: *op1,
                        args: args1
                            .iter()
                            .zip(args2)
                            .map(|(st1, st2)| TRS::lsc_helper(st1, st2, map))
                            .collect_vec(),
                    }
                }
            }
            _ => Context::Hole,
        }
    }
    // The workhorse behind generalization.
    fn generalize_clauses(
        lex: &Lexicon,
        lhs_context: &Context,
        rhs_context: &Context,
        clauses: &[Rule],
    ) -> Result<Vec<Rule>, SampleError> {
        // Create the LHS.
        let (lhs, lhs_place, var) =
            TRS::fill_hole_with_variable(lex, &lhs_context).ok_or(SampleError::Subterm)?;
        // Fill the RHS context and create subproblem rules.
        let mut rhs = rhs_context.clone();
        let mut new_rules: Vec<Rule> = vec![];
        for rhs_place in &rhs_context.holes() {
            // Collect term, type, and variable information from each clause.
            let (types, terms, vars) =
                TRS::collect_information(lex, &lhs, &lhs_place, rhs_place, clauses, var)?;
            // Infer the type for this place.
            let return_tp = TRS::compute_place_type(lex, &types)?;
            // Create the new operator for this place. TODO HACK: make applicative parameterizable.
            let new_op = TRS::new_operator(lex, true, &vars, &return_tp)?;
            // Create the rules expressing subproblems for this place.
            for (lhs_term, rhs_term) in &terms {
                let new_rule = TRS::new_rule(lex, new_op, lhs_term, rhs_term, var, &vars)?;
                new_rules.push(new_rule);
            }
            // Fill the hole at this place in the RHS.
            rhs = TRS::fill_next_hole(lex, &rhs, rhs_place, new_op, vars)?;
        }
        let rhs_term = rhs.to_term().map_err(|_| SampleError::Subterm)?;
        // Create the generalized rule.
        new_rules.push(Rule::new(lhs, vec![rhs_term]).ok_or(SampleError::Subterm)?);
        Ok(new_rules)
    }
    fn fill_next_hole(
        lex: &Lexicon,
        rhs: &Context,
        place: &[usize],
        new_op: Operator,
        vars: Vec<Variable>,
    ) -> Result<Context, SampleError> {
        let app = lex.has_op(Some("."), 2).map_err(|_| SampleError::Subterm)?;
        let mut subctx = Context::from(Atom::from(new_op));
        for var in vars {
            subctx = Context::Application {
                op: app,
                args: vec![subctx, Context::from(Atom::from(var))],
            };
        }
        rhs.replace(place, subctx).ok_or(SampleError::Subterm)
    }
    #[allow(clippy::type_complexity)]
    fn collect_information<'c>(
        lex: &Lexicon,
        lhs: &Term,
        lhs_place: &[usize],
        rhs_place: &[usize],
        clauses: &'c [Rule],
        var: Variable,
    ) -> Result<(Vec<Type>, Vec<(&'c Term, Term)>, Vec<Variable>), SampleError> {
        let mut terms = vec![];
        let mut types = vec![];
        let mut vars = vec![var];
        for clause in clauses {
            let rhs = clause.rhs().ok_or(SampleError::Subterm)?;
            let lhs_subterm = clause.lhs.at(lhs_place).ok_or(SampleError::Subterm)?;
            let rhs_subterm = rhs.at(rhs_place).ok_or(SampleError::Subterm)?;
            let mut map = HashMap::new();
            lex.infer_term(&rhs, &mut map).drop()?;
            types.push(map[rhs_place].clone());
            let alpha = Term::pmatch(vec![(&lhs, &clause.lhs)]).ok_or(SampleError::Subterm)?;
            for &var in &rhs_subterm.variables() {
                let var_term = Term::Variable(var);
                if let Some((&k, _)) = alpha.iter().find(|(_, &v)| *v == var_term) {
                    vars.push(k.clone())
                } else {
                    return Err(SampleError::Subterm);
                }
            }
            terms.push((lhs_subterm, rhs_subterm.clone()));
        }
        Ok((types, terms, vars))
    }
    fn compute_place_type(lex: &Lexicon, types: &[Type]) -> Result<Type, SampleError> {
        let return_tp = lex.fresh_type_variable();
        for tp in types {
            lex.unify(&return_tp, tp)
                .map_err(|_| SampleError::Subterm)?;
        }
        Ok(return_tp.apply(&lex.0.ctx.read().expect("poisoned context")))
    }
    // Patch a one-hole `Context` with a `Variable`.
    fn fill_hole_with_variable(
        lex: &Lexicon,
        context: &Context,
    ) -> Option<(Term, Vec<usize>, Variable)> {
        // Confirm that there's a single hole and find its place.
        let mut holes = context.holes();
        if holes.len() != 1 {
            return None;
        }
        let hole = holes.pop()?;
        // Create a variable whose type matches the hole.
        let mut tps = HashMap::new();
        lex.infer_context(context, &mut tps).drop().ok()?;
        let new_var = lex.invent_variable(&tps[&hole]);
        // Replace the hole with the new variable.
        let filled_context = context.replace(&hole, Context::from(Atom::from(new_var)))?;
        let term = filled_context.to_term().ok()?;
        // Return the term, the hole, and its replacement.
        Some((term, hole, new_var))
    }
    // Create a new `Operator` whose type is consistent with `Vars`.
    fn new_operator(
        lex: &Lexicon,
        applicative: bool,
        vars: &[Variable],
        return_tp: &Type,
    ) -> Result<Operator, SampleError> {
        // Construct the name.
        let name = None;
        // Construct the arity.
        let arity = (!applicative as u8) * (vars.len() as u8);
        // Construct the type.
        let mut tp = return_tp.clone();
        for &var in vars.iter().rev() {
            let schema = lex.infer_var(var)?;
            tp = Type::arrow(lex.instantiate(&schema), tp);
        }
        // Create the new variable.
        Ok(lex.invent_operator(name, arity, &tp))
    }
    // Create a new rule setting up a generalization subproblem.
    fn new_rule(
        lex: &Lexicon,
        op: Operator,
        lhs_arg: &Term,
        rhs: &Term,
        var: Variable,
        vars: &[Variable],
    ) -> Result<Rule, SampleError> {
        let mut lhs = Term::apply(op, vec![]).ok_or(SampleError::Subterm)?;
        let app = lex.has_op(Some("."), 2).map_err(|_| SampleError::Subterm)?;
        for &v in vars {
            let arg = if v == var {
                lhs_arg.clone()
            } else {
                Term::Variable(v)
            };
            lhs = Term::apply(app, vec![lhs, arg]).ok_or(SampleError::Subterm)?;
        }
        Rule::new(lhs, vec![rhs.clone()]).ok_or(SampleError::Subterm)
    }
}

#[cfg(test)]
mod tests {
    use super::TRS;
    use itertools::Itertools;
    use polytype::Context as TypeContext;
    use std::collections::HashMap;
    use term_rewriting::{Atom, Context};
    use trs::parser::{parse_context, parse_lexicon, parse_rule, parse_term, parse_trs};

    #[test]
    fn find_lhs_context_test() {
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
            "^(+(x_ 1) 2) = +(^(x_ 2) +(*(2 x_) 1)); ^(+(x_ 2) 2) = +(^(x_ 2) +(*(4 x_) 4)); ^(+(x_ 3) 2) = +(^(x_ 2) +(*(6 x_) 9)); +(x_ 0) = x_; +(0 x_) = x_;",
            &lex,
        )
            .expect("parsed trs");
        let clauses = trs.utrs.clauses();
        let (context, clauses) = TRS::find_lhs_context(&clauses).unwrap();
        let sig = &lex.0.read().expect("poisoned lexicon").signature;

        assert_eq!("^(+(x_, [!]), 2)", context.pretty(sig));
        assert_eq!(3, clauses.len());
    }

    #[test]
    fn find_rhs_context_test() {
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
        let clauses = vec![
            parse_rule("^(+(x_ 1) 2) = +(^(x_ 2) +(*(2 x_) 1))", &lex).expect("parsed rule 1"),
            parse_rule("^(+(x_ 2) 2) = +(^(x_ 2) +(*(4 x_) 4))", &lex).expect("parsed rule 2"),
            parse_rule("^(+(x_ 3) 2) = +(^(x_ 2) +(*(6 x_) 9))", &lex).expect("parsed rule 3"),
        ];
        let (context, clauses) = TRS::find_rhs_context(&clauses).unwrap();
        let sig = &lex.0.read().expect("poisoned lexicon").signature;

        assert_eq!("+(^(x_, 2), +(*([!], x_), [!]))", context.pretty(sig));
        assert_eq!(3, clauses.len());
    }

    #[test]
    fn fill_hole_with_variable_test() {
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
        let context = parse_context("^(+(x_ [!]) 2)", &lex).expect("parsed context");
        let (term, place, var) = TRS::fill_hole_with_variable(&lex, &context).unwrap();
        let tp = lex
            .infer_context(&Context::from(Atom::from(var)), &mut HashMap::new())
            .drop()
            .unwrap();
        let sig = &lex.0.read().expect("poisoned lexicon").signature;

        assert_eq!("^(+(x_, var1_), 2)", term.pretty(sig));
        assert_eq!(vec![0, 1], place);
        assert_eq!("INT", tp.to_string());
    }

    #[test]
    fn new_operator_test() {
        let lex = parse_lexicon(
            "+/0: INT -> INT -> INT; */0: INT -> INT -> INT; ^/0: INT -> INT -> INT; ./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
            "",
            "",
            true,
            TypeContext::default(),
        )
        .unwrap();
        let applicative = true;
        let vars = &[
            lex.invent_variable(&tp!(INT)),
            lex.invent_variable(&tp!(LIST)),
        ];
        let return_tp = tp!(LIST);
        let op = TRS::new_operator(&lex, applicative, vars, &return_tp).unwrap();
        let tp = lex
            .infer_context(&Context::from(Atom::from(op.clone())), &mut HashMap::new())
            .drop()
            .unwrap();
        let sig = &lex.0.read().expect("poisoned lexicon").signature;

        assert_eq!("op4", op.display(sig));
        assert_eq!(0, op.arity());
        assert_eq!("INT → LIST → LIST", tp.to_string());
    }

    #[test]
    fn new_rule_test() {
        let lex = parse_lexicon(
            &[
                "+/2: INT -> INT -> INT;",
                " */2: INT -> INT -> INT;",
                " ^/2: INT -> INT -> INT;",
                "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
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
        let op = lex.invent_operator(Some("F".to_string()), 0, &tp![@arrow[tp!(INT), tp!(INT)]]);
        let lhs_arg = parse_term("1", &lex).unwrap();
        let rhs = parse_term("2", &lex).unwrap();
        let var = lex.invent_variable(&tp![INT]);
        let vars = vec![var.clone()];
        let rule = TRS::new_rule(&lex, op, &lhs_arg, &rhs, var, &vars).unwrap();
        let sig = &lex.0.read().expect("poisoned lexicon").signature;

        assert_eq!("F 1 = 2", rule.pretty(sig));
    }

    #[test]
    fn generalize_clauses_test() {
        let lex = parse_lexicon(
            &[
                "+/2: INT -> INT -> INT;",
                " */2: INT -> INT -> INT;",
                " ^/2: INT -> INT -> INT;",
                "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
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
            "^(+(x_ 1) 2) = +(^(x_ 2) +(*(2 x_) 1)); ^(+(x_ 2) 2) = +(^(x_ 2) +(*(4 x_) 4)); ^(+(x_ 3) 2) = +(^(x_ 2) +(*(6 x_) 9)); +(x_ 0) = x_; +(0 x_) = x_;",
            &lex,
        )
            .expect("parsed trs");
        let all_clauses = trs.utrs.clauses();
        let (lhs_context, clauses) = TRS::find_lhs_context(&all_clauses).unwrap();
        let (rhs_context, clauses) = TRS::find_rhs_context(&clauses).unwrap();
        let rules =
            TRS::generalize_clauses(&trs.lex, &lhs_context, &rhs_context, &clauses).unwrap();
        let sig = &lex.0.read().expect("poisoned lexicon").signature;
        let rule_string = rules.iter().map(|r| r.pretty(sig)).join("\n");

        assert_eq!(
            "op11 1 = 2\nop11 2 = 4\nop11 3 = 6\nop12 1 = 1\nop12 2 = 4\nop12 3 = 9\n^(+(x_, var5_), 2) = +(^(x_, 2), +(*(op11 var5_, x_), op12 var5_))",
            rule_string
        );
    }

    #[test]
    fn generalize_test_1() {
        let lex = parse_lexicon(
            &[
                "+/2: int -> int -> int;",
                " */2: int -> int -> int;",
                " ^/2: int -> int -> int;",
                "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
                " 0/0: int; 1/0: int; 2/0: int;",
                " 3/0: int; 4/0: int; 6/0: int;",
                " 9/0: int;",
            ]
            .join(" "),
            "",
            "",
            true,
            TypeContext::default(),
        )
        .unwrap();
        let trs = parse_trs(
            "^(+(x_ 1) 2) = +(^(x_ 2) +(*(2 x_) 1)); ^(+(x_ 2) 2) = +(^(x_ 2) +(*(4 x_) 4)); ^(+(x_ 3) 2) = +(^(x_ 2) +(*(6 x_) 9)); +(x_ 0) = x_; +(0 x_) = x_;",
            &lex,
        )
            .expect("parsed trs");
        let trs = trs.generalize(&[]).unwrap().pop().unwrap();
        let sig = &lex.0.read().expect("poisoned lexicon").signature;
        let rule_string = trs.utrs.rules.iter().map(|r| r.pretty(sig)).join("\n");

        assert_eq!(
            "+(x_, 0) = x_\n+(0, x_) = x_\nop11 1 = 2\nop11 2 = 4\nop11 3 = 6\nop12 1 = 1\nop12 2 = 4\nop12 3 = 9\n^(+(x_, var5_), 2) = +(^(x_, 2), +(*(op11 var5_, x_), op12 var5_))",
            rule_string
        );
    }

    #[test]
    fn generalize_test_2() {
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
            "4 = 5; C (CONS (DIGIT 3) NIL) = (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 2) 5) NIL)); C (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 1) (CONS (DIGIT 3) NIL))) = (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 2) 5) NIL));",
            &lex,
        )
            .expect("parsed trs");
        let trs = trs.generalize(&[]).unwrap().pop().unwrap();
        let sig = &lex.0.read().expect("poisoned lexicon").signature;
        let rule_string = trs.utrs.rules.iter().map(|r| r.pretty(sig)).join("\n");

        assert_eq!(
            "4 = 5\nC var0_ = CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 2) 5) [])",
            rule_string
        );
    }
}

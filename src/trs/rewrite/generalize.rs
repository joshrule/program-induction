use super::{SampleError, TRS};
use itertools::Itertools;
use polytype::atype::{Schema, Ty, Type, Variable as TVar};
use std::{collections::HashMap, convert::TryFrom};
use term_rewriting::{Atom, Context, Operator, Rule, SituatedAtom, Term, Variable};
use trs::{Env, Lexicon};

impl<'ctx, 'b> TRS<'ctx, 'b> {
    pub fn generalize(&self) -> Result<Self, SampleError<'ctx>> {
        let (lhs_context, clauses) = TRS::find_lhs_context(&self.utrs.rules)?;
        let (rhs_context, clauses) = TRS::find_rhs_context(&clauses)?;
        let mut trs = self.clone();
        let mut new_rules = trs.generalize_clauses(&lhs_context, &rhs_context, &clauses)?;
        trs.remove_clauses(&clauses)?;
        let start = trs.len();
        trs.filter_background(&mut new_rules);
        trs.append_clauses(new_rules)?;
        let stop = trs.len();
        Ok(trs.smart_delete(start, stop)?)
    }
    fn find_lhs_context(clauses: &[Rule]) -> Result<(Context, Vec<Rule>), SampleError<'ctx>> {
        TRS::find_shared_context(clauses, |c| c.lhs.clone(), 1)
    }
    fn find_rhs_context(clauses: &[Rule]) -> Result<(Context, Vec<Rule>), SampleError<'ctx>> {
        TRS::find_shared_context(clauses, |c| c.rhs().unwrap(), 3) // TODO: constant is a HACK
    }
    fn find_shared_context<T>(
        clauses: &[Rule],
        f: T,
        max_holes: usize,
    ) -> Result<(Context, Vec<Rule>), SampleError<'ctx>>
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
        &mut self,
        lhs_context: &Context,
        rhs_context: &Context,
        clauses: &[Rule],
    ) -> Result<Vec<Rule>, SampleError<'ctx>> {
        // Create the LHS.
        let (lhs, lhs_place, var) =
            TRS::fill_hole_with_variable(&lhs_context).ok_or(SampleError::Subterm)?;
        let mut env = self.lex.infer_term(&lhs)?;
        env.invent = false;
        // Fill the RHS context and create subproblem rules.
        let mut rhs = rhs_context.clone();
        let mut new_rules: Vec<Rule> = vec![];
        for rhs_place in &rhs_context.holes() {
            // Collect term, type, and variable information from each clause.
            let (types, terms, vars) =
                TRS::collect_information(&self.lex, &lhs, &lhs_place, rhs_place, clauses, var)?;
            // Infer the type for this place.
            let return_tp = TRS::compute_place_type(&mut env, &types)?;
            // Create the new operator for this place. TODO HACK: make applicative parameterizable.
            let new_op = TRS::new_operator(&mut self.lex, true, &vars, return_tp, &env)?;
            // Create the rules expressing subproblems for this place.
            for (lhs_term, rhs_term) in &terms {
                let new_rule = TRS::new_rule(&self.lex, new_op, lhs_term, rhs_term, var, &vars)?;
                new_rules.push(new_rule);
            }
            // Fill the hole at this place in the RHS.
            rhs = TRS::fill_next_hole(&self.lex, &rhs, rhs_place, new_op, vars)?;
        }
        let rhs_term = Term::try_from(&rhs).map_err(|_| SampleError::Subterm)?;
        // Create the generalized rule.
        new_rules.push(Rule::new(lhs, vec![rhs_term]).ok_or(SampleError::Subterm)?);
        Ok(new_rules)
    }
    fn fill_next_hole(
        lex: &Lexicon<'ctx, 'b>,
        rhs: &Context,
        place: &[usize],
        new_op: Operator,
        vars: Vec<Variable>,
    ) -> Result<Context, SampleError<'ctx>> {
        let app = lex
            .has_operator(Some("."), 2)
            .map_err(|_| SampleError::Subterm)?;
        let mut subctx = Context::from(SituatedAtom::new(Atom::from(new_op), lex.signature()));
        for var in vars {
            subctx = Context::Application {
                op: app,
                args: vec![subctx, Context::from(var)],
            };
        }
        rhs.replace(place, subctx).ok_or(SampleError::Subterm)
    }
    #[allow(clippy::type_complexity)]
    fn collect_information<'c>(
        lex: &Lexicon<'ctx, 'b>,
        lhs: &Term,
        lhs_place: &[usize],
        rhs_place: &[usize],
        clauses: &'c [Rule],
        var: Variable,
    ) -> Result<(Vec<Ty<'ctx>>, Vec<(&'c Term, Term)>, Vec<Variable>), SampleError<'ctx>> {
        let mut terms = vec![];
        let mut types = vec![];
        let mut vars = vec![var];
        for clause in clauses {
            let rhs = clause.rhs().ok_or(SampleError::Subterm)?;
            let lhs_subterm = clause.lhs.at(lhs_place).ok_or(SampleError::Subterm)?;
            let rhs_subterm = rhs.at(rhs_place).ok_or(SampleError::Subterm)?;
            let env = lex.infer_term(&rhs)?;
            let map: HashMap<_, _> = rhs
                .subterms()
                .into_iter()
                .zip(&env.tps)
                .map(|((_, p), tp)| (p, *tp))
                .collect();
            types.push(map[rhs_place]);
            let alpha = Term::pmatch(&[(&lhs, &clause.lhs)]).ok_or(SampleError::Subterm)?;
            for &var in &rhs_subterm.variables() {
                let var_term = Term::Variable(var);
                if let Some((&k, _)) = alpha.0.iter().find(|(_, v)| **v == var_term) {
                    vars.push(k)
                } else {
                    return Err(SampleError::Subterm);
                }
            }
            terms.push((lhs_subterm, rhs_subterm.clone()));
        }
        Ok((types, terms, vars))
    }
    fn compute_place_type(
        env: &mut Env<'ctx, 'b>,
        types: &[Ty<'ctx>],
    ) -> Result<Ty<'ctx>, SampleError<'ctx>> {
        let tvar = TVar(env.src.fresh());
        let return_tp = env.lex.lex.ctx.intern_tvar(tvar);
        for tp in types {
            Type::unify_with_sub(&[(return_tp, tp)], &mut env.sub)
                .map_err(|_| SampleError::Subterm)?;
        }
        Ok(return_tp.apply(&env.sub))
    }
    // Patch a `Context` with a `Variable`.
    fn fill_hole_with_variable(context: &Context) -> Option<(Term, Vec<usize>, Variable)> {
        // Confirm that there's a hole and find its place.
        let hole = context.leftmost_hole()?;
        // Create a variable to fill the hole.
        let id = context.variables().len();
        let new_var = Variable(id);
        // Replace the hole with the new variable.
        let filled_context = context.replace(&hole, Context::from(new_var))?;
        let term = Term::try_from(&filled_context).ok()?;
        // Return the term, the hole, and its replacement.
        Some((term, hole, new_var))
    }
    // Create a new `Operator` whose type is consistent with `Vars`.
    fn new_operator(
        lex: &mut Lexicon<'ctx, 'b>,
        applicative: bool,
        vars: &[Variable],
        return_tp: Ty<'ctx>,
        env: &Env<'ctx, 'b>,
    ) -> Result<Operator, SampleError<'ctx>> {
        // Construct the name.
        let name = None;
        // Construct the arity.
        let arity = (!applicative as u8) * (vars.len() as u8);
        // Construct the type.
        let mut tp = return_tp;
        for &var in vars.iter().rev() {
            let schema: Schema<'ctx> = env.vars.get(var.0).ok_or(SampleError::Subterm)?;
            tp = env.lex.lex.ctx.arrow(
                schema.instantiate(&env.lex.lex.ctx, &mut lex.lex.to_mut().src),
                tp,
            );
        }
        tp = tp.apply(&env.sub);
        // Create the new variable.
        Ok(lex.invent_operator(name, arity, &tp))
    }
    // Create a new rule setting up a generalization subproblem.
    fn new_rule(
        lex: &Lexicon<'ctx, 'b>,
        op: Operator,
        lhs_arg: &Term,
        rhs: &Term,
        var: Variable,
        vars: &[Variable],
    ) -> Result<Rule, SampleError<'ctx>> {
        let mut lhs = Term::apply(op, vec![], lex.signature()).ok_or(SampleError::Subterm)?;
        let app = lex
            .has_operator(Some("."), 2)
            .map_err(|_| SampleError::Subterm)?;
        for &v in vars {
            let arg = if v == var {
                lhs_arg.clone()
            } else {
                Term::Variable(v)
            };
            lhs = Term::apply(app, vec![lhs, arg], lex.signature()).ok_or(SampleError::Subterm)?;
        }
        Rule::new(lhs, vec![rhs.clone()]).ok_or(SampleError::Subterm)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use polytype::atype::{with_ctx, TypeContext, Variable as TVar};
    use std::collections::HashMap;
    use term_rewriting::{Atom, Context, SituatedAtom, Variable};
    use trs::parser::{parse_context, parse_lexicon, parse_rule, parse_term, parse_trs};
    use trs::{Env, Lexicon, TRS};

    fn create_test_lexicon<'ctx, 'b>(ctx: &TypeContext<'ctx>) -> Lexicon<'ctx, 'b> {
        parse_lexicon(
            &[
                "+/0: INT -> INT -> INT;",
                " */0: INT -> INT -> INT;",
                " ^/0: INT -> INT -> INT;",
                "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
                " 0/0: INT; 1/0: INT; 2/0: INT;",
                " 3/0: INT; 4/0: INT; 6/0: INT;",
                " 9/0: INT;",
            ]
            .join(" "),
            ctx,
        )
        .unwrap()
    }
    fn create_list_test_lexicon<'ctx, 'b>(ctx: &TypeContext<'ctx>) -> Lexicon<'ctx, 'b> {
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
            ctx,
        )
        .unwrap()
    }

    #[test]
    fn find_lhs_context_test() {
        with_ctx(1024, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
            let clauses = vec![
                parse_rule("^ (+ x_ 1) 2 = + (^ x_ 2) (+ (* 2 x_) 1)", &mut lex).expect("rule 1"),
                parse_rule("^ (+ x_ 2) 2 = + (^ x_ 4) (+ (* 2 x_) 4)", &mut lex).expect("rule 2"),
                parse_rule("^ (+ x_ 3) 2 = + (^ x_ 6) (+ (* 2 x_) 9)", &mut lex).expect("rule 3"),
                parse_rule("+ x_ 0 = x_", &mut lex).expect("parsed rule 4"),
                parse_rule("+ 0 x_ = x_", &mut lex).expect("parsed rule 5"),
            ];
            let (context, clauses) = TRS::find_lhs_context(&clauses).unwrap();
            let sig = lex.signature();
            assert_eq!("^ (+ v0_ [!]) 2", context.pretty(sig));
            assert_eq!(3, clauses.len());
        })
    }

    #[test]
    fn find_rhs_context_test() {
        with_ctx(1024, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
            let clauses = vec![
                parse_rule("^ (+ x_ 1) 2 = + (^ x_ 2) (+ (* 2 x_) 1)", &mut lex)
                    .expect("parsed rule 1"),
                parse_rule("^ (+ x_ 2) 2 = + (^ x_ 4) (+ (* 2 x_) 4)", &mut lex)
                    .expect("parsed rule 2"),
                parse_rule("^ (+ x_ 3) 2 = + (^ x_ 6) (+ (* 2 x_) 9)", &mut lex)
                    .expect("parsed rule 3"),
            ];
            let (context, clauses) = TRS::find_rhs_context(&clauses).unwrap();
            let sig = lex.signature();
            assert_eq!("+ (^ v0_ [!]) (+ (* 2 v0_) [!])", context.pretty(sig));
            assert_eq!(3, clauses.len());
        })
    }

    #[test]
    fn fill_hole_with_variable_test() {
        with_ctx(1024, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
            let context = parse_context("^ (+ v0_ [!]) 2", &mut lex).expect("context");
            let (term, place, _) = TRS::fill_hole_with_variable(&context).unwrap();

            let env = lex.infer_term(&term).expect("env");
            let map: HashMap<_, _> = term
                .subterms()
                .into_iter()
                .zip(&env.tps)
                .map(|((_, p), tp)| (p, *tp))
                .collect();
            let tp = map[&place].apply(&env.sub);

            assert_eq!("^ (+ v0_ v1_) 2", term.pretty(lex.signature()));
            assert_eq!(vec![0, 1, 1], place);
            assert_eq!("INT", tp.to_string());
        })
    }

    #[test]
    fn new_operator_test() {
        with_ctx(1024, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
            let applicative = true;
            let vars = &[Variable(0), Variable(1)];
            let list = ctx.intern_name("list");
            let int = ctx.intern_name("int");
            let t_list = ctx.intern_tcon(list, &[]);
            let t_int = ctx.intern_tcon(int, &[]);
            let mut env = Env::new(true, &lex, Some(lex.lex.src));
            let dummy = env.src.fresh();
            env.new_variable(); // Add v0_.
            env.new_variable(); // Add v1_.
            env.sub.add(TVar(dummy + 1), t_int);
            env.sub.add(TVar(dummy + 2), t_list);
            let op = TRS::new_operator(&mut lex, applicative, vars, t_list, &env).expect("op");
            let context = Context::from(SituatedAtom::new(Atom::from(op), lex.signature()));
            let tp = lex
                .infer_context(&context)
                .map(|env| env.tps[0].apply(&env.sub))
                .expect("tp");
            assert_eq!(11, op.id());
            assert_eq!(0, op.arity(lex.signature()));
            assert_eq!("int → list → list", tp.to_string());
        })
    }

    #[test]
    fn new_rule_test() {
        with_ctx(1024, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
            let int = ctx.intern_name("int");
            let t_int = ctx.intern_tcon(int, &[]);
            let tp = ctx.arrow(t_int, t_int);
            let op = lex.invent_operator(Some("F".to_string()), 0, tp);
            let lhs_arg = parse_term("1", &mut lex).unwrap();
            let rhs = parse_term("2", &mut lex).unwrap();
            let var = Variable(0);
            let vars = vec![var];
            let rule = TRS::new_rule(&lex, op, &lhs_arg, &rhs, var, &vars).expect("rule");
            assert_eq!("F 1 = 2", rule.pretty(lex.signature()));
        })
    }

    #[test]
    fn generalize_clauses_test() {
        with_ctx(1024, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
            let mut trs = parse_trs(
                &[
                    "^ (+ x_ 1) 2 = + (^ x_ 2) (+ (* 2 x_) 1);",
                    "^ (+ x_ 2) 2 = + (^ x_ 2) (+ (* 4 x_) 4);",
                    "^ (+ x_ 3) 2 = + (^ x_ 2) (+ (* 6 x_) 9);",
                    "+ x_ 0 = x_;",
                    "+ 0 x_ = x_;",
                ]
                .join(" "),
                &mut lex,
                true,
                &[],
            )
            .expect("parsed trs");
            let (lhs_context, clauses) = TRS::find_lhs_context(&trs.utrs.rules).unwrap();
            let (rhs_context, clauses) = TRS::find_rhs_context(&clauses).unwrap();
            let rules = trs
                .generalize_clauses(&lhs_context, &rhs_context, &clauses)
                .unwrap();
            let sig = trs.lex.signature();
            let rule_string = rules.iter().map(|r| r.pretty(sig)).join("\n");
            let expected = [
                "op11 1 = 2",
                "op11 2 = 4",
                "op11 3 = 6",
                "op12 1 = 1",
                "op12 2 = 4",
                "op12 3 = 9",
                "^ (+ v0_ v1_) 2 = + (^ v0_ 2) (+ (* (op11 v1_) v0_) (op12 v1_))",
            ]
            .join("\n");
            assert_eq!(expected, rule_string);
        })
    }

    #[test]
    fn generalize_test_1() {
        with_ctx(1024, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
            let trs = parse_trs(
                &[
                    "^ (+ x_ 1) 2 = + (^ x_ 2) (+ (* 2 x_) 1);",
                    "^ (+ x_ 2) 2 = + (^ x_ 2) (+ (* 4 x_) 4);",
                    "^ (+ x_ 3) 2 = + (^ x_ 2) (+ (* 6 x_) 9);",
                    "+ x_ 0 = x_;",
                    "+ 0 x_ = x_;",
                ]
                .join(" "),
                &mut lex,
                true,
                &[],
            )
            .expect("parsed trs");
            let trs = trs.generalize().unwrap();
            let sig = trs.lex.signature();
            let trs_string = trs.utrs.rules.iter().map(|r| r.pretty(sig)).join("\n");
            let expected = [
                "+ v0_ 0 = v0_",
                "+ 0 v0_ = v0_",
                "op11 1 = 2",
                "op11 2 = 4",
                "op11 3 = 6",
                "op12 1 = 1",
                "op12 2 = 4",
                "op12 3 = 9",
                "^ (+ v0_ v1_) 2 = + (^ v0_ 2) (+ (* (op11 v1_) v0_) (op12 v1_))",
            ]
            .join("\n");

            assert_eq!(expected, trs_string);
        })
    }

    #[test]
    fn generalize_test_2() {
        with_ctx(1024, |ctx| {
            let mut lex = create_list_test_lexicon(&ctx);
            let trs = parse_trs(
                &[
                    "4 = 5;",
                    "C (CONS (DIGIT 3) NIL) = (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 2) 5) NIL));",
                    "C (CONS (DIGIT 9) (CONS (DECC (DIGIT 1) 1) (CONS (DIGIT 3) NIL))) = (CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 2) 5) NIL));",
                ].join(""),
                &mut lex,
                true,
                &[],
            )
                .expect("parsed trs");
            let trs = trs.generalize().unwrap();
            let sig = trs.lex.signature();
            let trs_string = trs.utrs.rules.iter().map(|r| r.pretty(sig)).join("\n");

            assert_eq!(
                "4 = 5\nC v0_ = CONS (DECC (DIGIT 1) 6) (CONS (DECC (DIGIT 2) 5) [])",
                trs_string
            );
        })
    }
}

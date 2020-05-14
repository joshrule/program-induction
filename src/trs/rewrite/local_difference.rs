use super::{super::as_result, SampleError, TRS};
use itertools::Itertools;
use rand::Rng;
use std::iter::once;
use term_rewriting::{Rule, Term};

impl<'ctx, 'b> TRS<'ctx, 'b> {
    /// Selects a rule from the TRS at random, finds all differences in the LHS and RHS,
    /// and makes rules from those differences and inserts them back into the TRS imediately after the background.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # use programinduction::trs::{parse_lexicon, parse_trs};
    /// # use rand::thread_rng;
    /// # use polytype::{Source, atype::{with_ctx, TypeSchema, TypeContext}};
    /// with_ctx(10, |ctx: TypeContext<'_>| {
    ///     let mut rng = thread_rng();
    ///     let mut lex = parse_lexicon(
    ///         "PLUS/2: int -> int -> int; SUCC/1: int -> int; ZERO/0: int;",
    ///         &ctx,
    ///     ).expect("lex");
    ///
    ///     let trs = parse_trs(
    ///         "SUCC(PLUS(v0_ SUCC(v1_))) = SUCC(SUCC(PLUS(v0_ v1_)));",
    ///         &mut lex, true, &[],
    ///     ).expect("trs");
    ///     assert_eq!(1, trs.len());
    ///
    ///     let new_trss = trs.local_difference(&mut rng).unwrap();
    ///     assert_eq!(2, new_trss.len())
    /// })
    /// ```
    pub fn local_difference<R: Rng>(&self, rng: &mut R) -> Result<Vec<Self>, SampleError<'ctx>> {
        let (n, clause) = self.choose_clause(rng)?;
        let mut new_rules = TRS::local_difference_helper(&clause);
        self.filter_background(&mut new_rules);
        let new_trss = new_rules
            .into_iter()
            .filter_map(|r| {
                let mut trs = self.clone();
                trs.replace(n, &clause, r).ok()?;
                Some(trs)
            })
            .collect_vec();
        as_result(new_trss)
    }
    /// Given a rule that has similar terms in the lhs and rhs,
    /// returns a list of rules where each similarity is removed one at a time
    fn local_difference_helper(rule: &Rule) -> Vec<Rule> {
        if let Some(rhs) = rule.rhs() {
            TRS::find_differences(&rule.lhs, &rhs)
                .into_iter()
                .filter_map(|(lhs, rhs)| Rule::new(lhs, vec![rhs]))
                .collect_vec()
        } else {
            vec![]
        }
    }
    // helper for local difference, finds differences in the given lhs and rhs recursively
    fn find_differences(lhs: &Term, rhs: &Term) -> Vec<(Term, Term)> {
        if lhs == rhs {
            return vec![];
        }
        match (lhs, rhs) {
            (Term::Variable(_), _) => vec![], // Variable can't be head of rule
            (
                Term::Application {
                    op: lop,
                    args: largs,
                },
                Term::Application {
                    op: rop,
                    args: rargs,
                },
            ) if lop == rop && !largs.is_empty() => largs
                .iter()
                .zip(rargs)
                .flat_map(|(l, r)| TRS::find_differences(l, r))
                .chain(once((lhs.clone(), rhs.clone())))
                .collect_vec(),
            _ => vec![(lhs.clone(), rhs.clone())],
        }
    }
}

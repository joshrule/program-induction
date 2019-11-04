use itertools::Itertools;
use rand::Rng;
use std::iter::once;
use term_rewriting::{Rule, Term};

use super::{SampleError, TRS};

impl TRS {
    /// Selects a rule from the TRS at random, finds all differences in the LHS and RHS,
    /// and makes rules from those differences and inserts them back into the TRS imediately after the background.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use polytype::Context as TypeContext;
    /// # use rand::{thread_rng};
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule};
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some(".".to_string()));
    /// ops.push(ptp![0, 1; @arrow[tp!(@arrow[tp!(0), tp!(1)]), tp!(0), tp!(1)]]);
    /// sig.new_op(2, Some("PLUS".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// sig.new_op(1, Some("SUCC".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    /// sig.new_op(0, Some("ZERO".to_string()));
    /// ops.push(ptp![int]);
    ///
    /// let rules = vec![
    ///     parse_rule(&mut sig, "SUCC(PLUS(x_ SUCC(y_))) = SUCC(SUCC(PLUS(x_ y_)))").expect("parsed rule"),
    /// ];
    ///
    /// let vars = vec![
    ///     ptp![int],
    ///     ptp![int],
    ///     ptp![int],
    /// ];
    ///
    /// for op in sig.operators() {
    ///     println!("{:?}/{}", op.name(&sig), op.arity())
    /// }
    /// for r in &rules {
    ///     println!("{:?}", r.pretty(&sig));
    /// }
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], vec![], false, TypeContext::default());
    ///
    /// let mut trs = TRS::new(&lexicon, rules).unwrap();
    ///
    /// assert_eq!(trs.len(), 1);
    ///
    /// let mut rng = thread_rng();
    ///
    /// let num_new_trss = trs.local_difference(&mut rng).map(|x| x.len()).ok();
    ///
    /// assert_eq!(Some(2), num_new_trss)
    /// # }
    /// ```
    pub fn local_difference<R: Rng>(&self, rng: &mut R) -> Result<Vec<TRS>, SampleError> {
        let clause = self.choose_clause(rng)?;
        let new_rules = TRS::local_difference_helper(&clause);
        let new_trss = new_rules
            .into_iter()
            .filter_map(|r| {
                let mut trs = self.clone();
                if trs.utrs.replace(0, &clause, r).is_ok() {
                    Some(trs)
                } else {
                    None
                }
            })
            .collect_vec();
        if new_trss.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(new_trss)
        }
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

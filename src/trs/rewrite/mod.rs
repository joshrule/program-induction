//! Polymorphically typed (Hindley-Milner) First-Order Term Rewriting Systems (no abstraction)
//!
//! Much thanks to:
//! - https://github.com/rob-smallshire/hindley-milner-python
//! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! - (TAPL; Pierce, 2002, ch. 22)

mod add_exception;
mod delete_rule;
mod generalize;
mod local_difference;
mod log_likelihood;
mod log_posterior;
mod log_prior;
mod move_rule;
mod recurse;
mod regenerate_rule;
mod sample_rule;
mod swap_lhs_and_rhs;
mod variablize;

use itertools::Itertools;
use rand::{seq::IteratorRandom, Rng};
use std::collections::HashMap;
use std::fmt;
use term_rewriting::{Rule, TRS as UntypedTRS};

use super::{Lexicon, Likelihood, ModelParams, Prior, SampleError, TypeError};

/// Manages the semantics of a term rewriting system.
#[derive(Debug, PartialEq, Clone)]
pub struct TRS {
    pub(crate) lex: Lexicon,
    // INVARIANT: UntypedTRS.rules ends with lex.background
    pub(crate) utrs: UntypedTRS,
}
impl TRS {
    /// Create a new `TRS` under the given [`Lexicon`]. Any background knowledge
    /// will be appended to the given ruleset.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use term_rewriting::{Signature, parse_rule};
    /// # use polytype::Context as TypeContext;
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some("PLUS".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// sig.new_op(1, Some("SUCC".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    /// sig.new_op(0, Some("ZERO".to_string()));
    /// ops.push(ptp![int]);
    ///
    /// let rules = vec![
    ///     parse_rule(&mut sig, "PLUS(x_ ZERO) = x_").expect("parsed rule"),
    ///     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))").expect("parsed rule"),
    /// ];
    ///
    /// let vars = vec![
    ///     ptp![int],
    ///     ptp![int],
    ///     ptp![int],
    /// ];
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], vec![], false, TypeContext::default());
    ///
    /// let ctx = lexicon.context();
    ///
    /// let trs = TRS::new(&lexicon, rules).unwrap();
    ///
    /// assert_eq!(trs.size(), 12);
    /// # }
    /// ```
    /// [`Lexicon`]: struct.Lexicon.html
    pub fn new(lexicon: &Lexicon, mut rules: Vec<Rule>) -> Result<TRS, TypeError> {
        let utrs = {
            let utrs = {
                let lex = lexicon.0.read().expect("poisoned lexicon");
                rules.append(&mut lex.background.clone());
                let mut utrs = UntypedTRS::new(rules);
                if lexicon.0.read().expect("poisoned lexicon").deterministic {
                    utrs.make_deterministic();
                }
                utrs
            };
            lexicon.infer_utrs(&utrs)?;
            utrs
        };
        Ok(TRS {
            lex: lexicon.clone(),
            utrs,
        })
    }

    pub fn lexicon(&self) -> Lexicon {
        self.lex.clone()
    }

    /// The size of the underlying [`term_rewriting::TRS`].
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn size(&self) -> usize {
        self.utrs.size()
    }

    /// The length of the underlying [`term_rewriting::TRS`].
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn len(&self) -> usize {
        self.utrs.len()
    }

    /// Is the underlying [`term_rewriting::TRS`] empty?.
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn is_empty(&self) -> bool {
        self.utrs.is_empty()
    }

    /// The underlying [`term_rewriting::TRS`].
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn utrs(&self) -> UntypedTRS {
        self.utrs.clone()
    }

    pub fn num_background_rules(&self) -> usize {
        self.lex
            .0
            .read()
            .expect("poisoned lexicon")
            .background
            .len()
    }

    pub fn num_learned_rules(&self) -> usize {
        self.len() - self.num_background_rules()
    }

    pub fn replace(
        &mut self,
        n: usize,
        old_clause: &Rule,
        new_clause: Rule,
    ) -> Result<&mut TRS, SampleError> {
        self.lex
            .infer_rule(&new_clause, &mut HashMap::new())
            .drop()?;
        self.utrs.replace(n, old_clause, new_clause)?;
        Ok(self)
    }

    /// pick a single clause
    fn choose_clause<R: Rng>(&self, rng: &mut R) -> Result<(usize, Rule), SampleError> {
        let rules = {
            let num_rules = self.len();
            let num_background = self.num_background_rules();
            &self.utrs.rules[0..(num_rules - num_background)]
        };
        let mut clauses = rules
            .iter()
            .enumerate()
            .flat_map(|(i, rule)| rule.clauses().into_iter().map(move |r| (i, r)))
            .collect_vec();
        let idx = (0..clauses.len())
            .choose(rng)
            .ok_or(SampleError::OptionsExhausted)?;
        Ok(clauses.swap_remove(idx))
    }

    fn novel_rules(&self, rules: &[Rule]) -> Vec<Rule> {
        rules
            .iter()
            .cloned()
            .filter(|r| !self.contains(r))
            .collect()
    }

    fn clauses_for_learning(&self, data: &[Rule]) -> Result<Vec<Rule>, SampleError> {
        let n_rules = self.num_learned_rules();
        let mut all_rules = self.utrs.rules[..n_rules]
            .iter()
            .flat_map(Rule::clauses)
            .collect_vec();
        all_rules.extend_from_slice(&self.novel_rules(data));
        if all_rules.len() == 0 {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(all_rules)
        }
    }

    fn contains(&self, rule: &Rule) -> bool {
        self.utrs.get_clause(rule).is_some()
    }

    fn remove_clauses(&mut self, rules: &[Rule]) -> Result<(), SampleError> {
        for rule in rules {
            if self.contains(rule) {
                self.utrs.remove_clauses(&rule)?;
            }
        }
        Ok(())
    }

    fn prepend_clauses(&mut self, rules: Vec<Rule>) -> Result<(), SampleError> {
        self.utrs
            .pushes(rules)
            .map(|_| ())
            .map_err(SampleError::from)
    }
}
impl fmt::Display for TRS {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let true_len = self.num_learned_rules();
        let sig = &self.lex.0.read().expect("poisoned lexicon").signature;
        let trs_str = self
            .utrs
            .rules
            .iter()
            .take(true_len)
            .map(|r| format!("{};", r.display(sig)))
            .join("\n");

        write!(f, "{}", trs_str)
    }
}

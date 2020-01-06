//! Polymorphically typed (Hindley-Milner) First-Order Term Rewriting Systems (no abstraction)
//!
//! Much thanks to:
//! - https://github.com/rob-smallshire/hindley-milner-python
//! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! - (TAPL; Pierce, 2002, ch. 22)

mod combine;
mod compose;
mod delete_rule;
mod generalize;
mod local_difference;
mod log_likelihood;
mod log_posterior;
mod log_prior;
mod memorize;
mod move_rule;
mod recurse;
mod regenerate_rule;
mod sample_rule;
mod swap_lhs_and_rhs;
mod variablize;

use super::{Lexicon, Likelihood, ModelParams, Prior, SampleError, TypeError};
use gp::Tournament;
use itertools::Itertools;
use rand::{distributions::Distribution, seq::IteratorRandom, Rng};
use std::collections::HashMap;
use std::fmt;
use term_rewriting::{Rule, TRS as UntypedTRS};

pub type TRSMoves = Vec<WeightedTRSMove>;

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct WeightedTRSMove {
    pub weight: usize,
    pub mv: TRSMove,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TRSMoveName {
    Memorize,
    SampleRule,
    RegenerateRule,
    LocalDifference,
    MemorizeOne,
    DeleteRule,
    Variablize,
    Generalize,
    Recurse,
    RecurseVariablize,
    RecurseGeneralize,
    DeleteRules,
    Combine,
    Compose,
    ComposeVariablize,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum TRSMove {
    Memorize,
    SampleRule((f64, f64, f64, f64), usize),
    RegenerateRule((f64, f64, f64, f64), usize),
    LocalDifference,
    MemorizeOne,
    DeleteRule,
    Variablize,
    Generalize,
    Recurse(usize),
    RecurseVariablize(usize),
    RecurseGeneralize(usize),
    DeleteRules(usize),
    Combine(usize),
    Compose,
    ComposeVariablize,
}
impl TRSMove {
    pub fn take<R: Rng>(
        &self,
        lex: &Lexicon,
        bg: &[Rule],
        obs: &[Rule],
        rng: &mut R,
        parents: &[TRS],
    ) -> Result<Vec<TRS>, SampleError> {
        match *self {
            TRSMove::Memorize => Ok(TRS::memorize(lex, bg.to_vec(), obs)),
            TRSMove::SampleRule(aw, mss) => parents[0].sample_rule(aw, mss, rng),
            TRSMove::RegenerateRule(aw, mss) => parents[0].sample_rule(aw, mss, rng),
            TRSMove::LocalDifference => parents[0].local_difference(rng),
            TRSMove::MemorizeOne => parents[0].memorize_one(obs),
            TRSMove::DeleteRule => parents[0].delete_rule(),
            TRSMove::Variablize => parents[0].variablize(),
            TRSMove::Generalize => parents[0].generalize(&[]),
            TRSMove::Recurse(n) => parents[0].recurse(n),
            TRSMove::RecurseVariablize(n) => parents[0].recurse_and_variablize(n, rng),
            TRSMove::RecurseGeneralize(n) => parents[0].recurse_and_generalize(n, rng),
            TRSMove::DeleteRules(t) => parents[0].delete_rules(rng, t),
            TRSMove::Compose => parents[0].compose(),
            TRSMove::ComposeVariablize => parents[0].compose_and_variablize(rng),
            TRSMove::Combine(t) => TRS::combine(&parents[0], &parents[1], rng, t),
        }
    }
    pub fn get_parents<R: Rng>(&self, t: &Tournament<TRS>, rng: &mut R) -> Vec<TRS> {
        match *self {
            TRSMove::Memorize => vec![],
            TRSMove::Combine(_) => vec![t.sample(rng).clone(), t.sample(rng).clone()],
            _ => vec![t.sample(rng).clone()],
        }
    }
    pub(crate) fn name(&self) -> TRSMoveName {
        match *self {
            TRSMove::Memorize => TRSMoveName::Memorize,
            TRSMove::SampleRule(..) => TRSMoveName::SampleRule,
            TRSMove::RegenerateRule(..) => TRSMoveName::RegenerateRule,
            TRSMove::LocalDifference => TRSMoveName::LocalDifference,
            TRSMove::MemorizeOne => TRSMoveName::MemorizeOne,
            TRSMove::DeleteRule => TRSMoveName::DeleteRule,
            TRSMove::Variablize => TRSMoveName::Variablize,
            TRSMove::Generalize => TRSMoveName::Generalize,
            TRSMove::Recurse(..) => TRSMoveName::Recurse,
            TRSMove::RecurseVariablize(..) => TRSMoveName::RecurseVariablize,
            TRSMove::RecurseGeneralize(..) => TRSMoveName::RecurseGeneralize,
            TRSMove::DeleteRules(..) => TRSMoveName::DeleteRules,
            TRSMove::Combine(..) => TRSMoveName::Combine,
            TRSMove::Compose => TRSMoveName::Compose,
            TRSMove::ComposeVariablize => TRSMoveName::ComposeVariablize,
        }
    }
}

/// A typed term rewriting system.
#[derive(Debug, PartialEq, Clone)]
pub struct TRS {
    pub(crate) lex: Lexicon,
    // INVARIANT: utrs never contains background information
    pub(crate) background: Vec<Rule>,
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
    /// ```
    ///
    /// [`Lexicon`]: struct.Lexicon.html
    pub fn new(
        lexicon: &Lexicon,
        background: Vec<Rule>,
        rules: Vec<Rule>,
    ) -> Result<TRS, TypeError> {
        let trs = TRS::new_unchecked(lexicon, background, rules);
        lexicon.infer_utrs(&trs.utrs)?;
        Ok(trs)
    }

    /// Like [`TRS::new`] but skips type inference. This is useful in scenarios
    /// where you are already confident in the type safety of the new rules.
    ///
    /// [`TRS::new`]: struct.TRS.html#method.new
    pub fn new_unchecked(lexicon: &Lexicon, background: Vec<Rule>, rules: Vec<Rule>) -> TRS {
        // Remove any rules already in the background
        let mut utrs = UntypedTRS::new(rules);
        for bg in background.iter().flat_map(|r| r.clauses()) {
            utrs.remove_clauses(&bg).ok();
        }
        if lexicon.is_deterministic() {
            utrs.make_deterministic();
        }
        let lex = lexicon.clone();
        TRS {
            lex,
            background,
            utrs,
        }
    }

    pub fn lexicon(&self) -> Lexicon {
        self.lex.clone()
    }

    /// The size of the underlying [`term_rewriting::TRS`].
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn size(&self) -> usize {
        self.utrs.rules.iter().map(Rule::size).sum()
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

    pub fn full_utrs(&self) -> UntypedTRS {
        let mut utrs = self.utrs.clone();
        self.lex.filter_background(&mut utrs.rules);
        utrs.rules
            .extend_from_slice(&self.lex.0.read().expect("poisoned lexicon").background);
        utrs
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
        self.len()
    }

    pub fn replace(
        &mut self,
        n: usize,
        old_clause: &Rule,
        new_clause: Rule,
    ) -> Result<&mut TRS, SampleError> {
        // TODO: why are we type-checking here?
        self.lex
            .infer_rule(&new_clause, &mut HashMap::new())
            .drop()?;
        self.utrs.replace(n, old_clause, new_clause)?;
        Ok(self)
    }

    pub fn swap_rules(&mut self, rules: &[(Rule, &Rule)]) -> Result<&mut TRS, SampleError> {
        for (new_rule, old_rule) in rules {
            self.swap(old_rule, new_rule.clone())?;
        }
        Ok(self)
    }

    pub fn swap(&mut self, old_rule: &Rule, new_rule: Rule) -> Result<&mut TRS, SampleError> {
        if let Some((n, _)) = self.utrs.get_clause(old_rule) {
            self.utrs.replace(n, old_rule, new_rule)?;
        } else {
            self.utrs.insert(0, new_rule)?;
        }
        Ok(self)
    }

    pub fn is_alpha(trs1: &TRS, trs2: &TRS) -> bool {
        let m = trs1.num_learned_rules();
        let n = trs2.num_learned_rules();
        trs1.lex == trs2.lex
            && m == n
            && trs1.utrs.rules[..m]
                .iter()
                .zip(&trs2.utrs.rules[..m])
                .all(|(r1, r2)| Rule::alpha(r1, r2).is_some())
    }

    fn clauses(&self) -> Vec<(usize, Rule)> {
        self.utrs
            .rules
            .iter()
            .enumerate()
            .flat_map(|(i, rule)| rule.clauses().into_iter().map(move |r| (i, r)))
            .collect_vec()
    }

    /// pick a single clause
    fn choose_clause<R: Rng>(&self, rng: &mut R) -> Result<(usize, Rule), SampleError> {
        let mut clauses = self.clauses();
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
        let mut all_rules = self.utrs.rules.iter().flat_map(Rule::clauses).collect_vec();
        if !data.is_empty() {
            all_rules.extend_from_slice(&self.novel_rules(data));
        }
        if all_rules.is_empty() {
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

    fn append_clauses(&mut self, rules: Vec<Rule>) -> Result<(), SampleError> {
        self.utrs
            .inserts_idx(self.num_learned_rules(), rules)
            .map(|_| ())
            .map_err(SampleError::from)
    }
}
impl fmt::Display for TRS {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let sig = &self.lex.0.read().expect("poisoned lexicon").signature;
        let trs_str = self
            .utrs
            .rules
            .iter()
            .map(|r| format!("{};", r.display(sig)))
            .join("\n");

        write!(f, "{}", trs_str)
    }
}

//! Polymorphically typed (Hindley-Milner) First-Order Term Rewriting Systems (no abstraction)
//!
//! Much thanks to:
//! - https://github.com/rob-smallshire/hindley-milner-python
//! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! - (TAPL; Pierce, 2002, ch. 22)

mod combine;
mod compose;
mod delete_rule;
// mod generalize;
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
use term_rewriting::{MergeStrategy, Rule, RuleContext, Term, TRS as UntypedTRS};

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
    // Variablize,
    // Generalize,
    // Recurse,
    // RecurseVariablize,
    // RecurseGeneralize,
    DeleteRules,
    Combine,
    Compose,
    // ComposeVariablize,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum TRSMove {
    Memorize,
    SampleRule((f64, f64, f64, f64), usize),
    RegenerateRule((f64, f64, f64, f64), usize),
    LocalDifference,
    MemorizeOne,
    DeleteRule,
    // Variablize,
    // Generalize,
    // Recurse(usize),
    // RecurseVariablize(usize),
    // RecurseGeneralize(usize),
    DeleteRules(usize),
    Combine(usize),
    Compose,
    // ComposeVariablize,
}
impl TRSMove {
    pub fn take<'a, 'b, R: Rng>(
        &self,
        lex: &Lexicon<'b>,
        deterministic: bool,
        bg: &'a [Rule],
        contexts: &[RuleContext],
        obs: &[Rule],
        rng: &mut R,
        parents: &[TRS<'a, 'b>],
    ) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        match *self {
            TRSMove::Memorize => Ok(TRS::memorize(lex, deterministic, bg, obs)),
            TRSMove::SampleRule(aw, mss) => parents[0].sample_rule(contexts, aw, mss, rng),
            TRSMove::RegenerateRule(aw, mss) => parents[0].regenerate_rule(aw, mss, rng),
            TRSMove::LocalDifference => parents[0].local_difference(rng),
            TRSMove::MemorizeOne => parents[0].memorize_one(obs),
            TRSMove::DeleteRule => parents[0].delete_rule(),
            // TRSMove::Variablize => parents[0].variablize(),
            // TRSMove::Generalize => parents[0].generalize(&[]),
            // TRSMove::Recurse(n) => parents[0].recurse(n),
            // TRSMove::RecurseVariablize(n) => parents[0].recurse_and_variablize(n, rng),
            // TRSMove::RecurseGeneralize(n) => parents[0].recurse_and_generalize(n, rng),
            TRSMove::DeleteRules(t) => parents[0].delete_rules(rng, t),
            TRSMove::Compose => parents[0].compose(),
            // TRSMove::ComposeVariablize => parents[0].compose_and_variablize(rng),
            TRSMove::Combine(t) => TRS::combine(&parents[0], &parents[1], rng, t),
        }
    }
    pub fn get_parents<'a, 'b, R: Rng>(
        &self,
        t: &Tournament<TRS<'a, 'b>>,
        rng: &mut R,
    ) -> Vec<TRS<'a, 'b>> {
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
            // TRSMove::Variablize => TRSMoveName::Variablize,
            // TRSMove::Generalize => TRSMoveName::Generalize,
            // TRSMove::Recurse(..) => TRSMoveName::Recurse,
            // TRSMove::RecurseVariablize(..) => TRSMoveName::RecurseVariablize,
            // TRSMove::RecurseGeneralize(..) => TRSMoveName::RecurseGeneralize,
            TRSMove::DeleteRules(..) => TRSMoveName::DeleteRules,
            TRSMove::Combine(..) => TRSMoveName::Combine,
            TRSMove::Compose => TRSMoveName::Compose,
            // TRSMove::ComposeVariablize => TRSMoveName::ComposeVariablize,
        }
    }
}

/// A typed term rewriting system.
#[derive(Debug, PartialEq, Clone)]
pub struct TRS<'a, 'b> {
    pub(crate) lex: Lexicon<'b>,
    // INVARIANT: utrs never contains background information
    pub(crate) background: &'a [Rule],
    pub(crate) utrs: UntypedTRS,
}
impl<'a, 'b> TRS<'a, 'b> {
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
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, TypeContext::default());
    ///
    /// let ctx = lexicon.context();
    ///
    /// let trs = TRS::new(&lexicon, true, &[], rules).unwrap();
    ///
    /// assert_eq!(trs.size(), 12);
    /// ```
    ///
    /// [`Lexicon`]: struct.Lexicon.html
    pub fn new<'c, 'd>(
        lexicon: &Lexicon<'d>,
        deterministic: bool,
        background: &'c [Rule],
        rules: Vec<Rule>,
    ) -> Result<TRS<'c, 'd>, TypeError> {
        let trs = TRS::new_unchecked(lexicon, deterministic, background, rules);
        lexicon.infer_utrs(&trs.utrs)?;
        Ok(trs)
    }

    /// Like [`TRS::new`] but skips type inference. This is useful in scenarios
    /// where you are already confident in the type safety of the new rules.
    ///
    /// [`TRS::new`]: struct.TRS.html#method.new
    pub fn new_unchecked<'c, 'd>(
        lexicon: &Lexicon<'d>,
        deterministic: bool,
        background: &'c [Rule],
        rules: Vec<Rule>,
    ) -> TRS<'c, 'd> {
        // Remove any rules already in the background
        let mut utrs = UntypedTRS::new(rules);
        for bg in background.iter().flat_map(|r| r.clauses()) {
            utrs.remove_clauses(&bg).ok();
        }
        if deterministic {
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
        self.filter_background(&mut utrs.rules);
        utrs.rules.extend_from_slice(&self.background);
        utrs
    }

    pub fn num_background_rules(&self) -> usize {
        self.background.len()
    }

    pub fn num_learned_rules(&self) -> usize {
        self.len()
    }

    pub fn replace(
        &mut self,
        n: usize,
        old_clause: &Rule,
        new_clause: Rule,
    ) -> Result<&mut TRS<'a, 'b>, SampleError> {
        // TODO: why are we type-checking here?
        self.lex
            .infer_rule(&new_clause, &mut HashMap::new())
            .drop()?;
        self.utrs.replace(n, old_clause, new_clause)?;
        Ok(self)
    }

    pub fn swap_rules(&mut self, rules: &[(Rule, Rule)]) -> Result<&mut TRS<'a, 'b>, SampleError> {
        for (new_rule, old_rule) in rules {
            self.swap(&old_rule, new_rule.clone())?;
        }
        Ok(self)
    }

    pub fn swap(
        &mut self,
        old_rule: &Rule,
        new_rule: Rule,
    ) -> Result<&mut TRS<'a, 'b>, SampleError> {
        if let Some((n, _)) = self.utrs.get_clause(old_rule) {
            self.utrs.replace(n, old_rule, new_rule)?;
        } else {
            self.utrs.insert(0, new_rule)?;
        }
        Ok(self)
    }

    pub fn is_alpha(trs1: &TRS, trs2: &TRS) -> bool {
        if trs1.len() != trs2.len() || trs1.background != trs2.background {
            return false;
        }
        if let Ok((lex, sig_change)) =
            Lexicon::merge(&trs1.lex, &trs2.lex, MergeStrategy::OperatorsByArityAndName)
        {
            let reified_trs2 = sig_change.reify_trs(&lex.0.signature, trs2.utrs());
            trs1.utrs
                .rules
                .iter()
                .zip(&reified_trs2.rules)
                .all(|(r1, r2)| Rule::alpha(r1, r2).is_some())
        } else {
            false
        }
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
    pub fn filter_background(&self, rules: &mut Vec<Rule>) {
        for rule in rules.iter_mut() {
            for bg in self.background {
                rule.discard(bg);
            }
        }
        if self.utrs.is_deterministic() {
            rules.retain(|rule| {
                self.background
                    .iter()
                    .all(|bg| Term::alpha(vec![(&bg.lhs, &rule.lhs)]).is_none())
            });
        }
        rules.retain(|rule| !rule.is_empty());
    }
    /// Is the `Lexicon` deterministic?
    pub fn is_deterministic(&self) -> bool {
        self.utrs.is_deterministic()
    }
}
impl<'a, 'b> fmt::Display for TRS<'a, 'b> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let trs_str = self
            .utrs
            .rules
            .iter()
            .map(|r| format!("{};", r.display(&self.lex.0.signature)))
            .join("\n");

        write!(f, "{}", trs_str)
    }
}

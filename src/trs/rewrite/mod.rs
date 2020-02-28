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
mod meta;
mod move_rule;
mod recurse;
mod regenerate_rule;
mod sample_rule;
mod swap_lhs_and_rhs;
mod variablize;

use gp::{GPParams, Tournament};
use itertools::Itertools;
use rand::{distributions::Distribution, seq::IteratorRandom, Rng};
use std::{borrow::Borrow, collections::HashMap, fmt};
use term_rewriting::{MergeStrategy, Operator, Rule, Term, Variable, TRS as UntypedTRS};
use trs::{GeneticParamsFull, Lexicon, Prior, SampleError, TypeError, TRSGP};
use Task;

pub type TRSMoves = Vec<WeightedTRSMove>;
pub(crate) type Rules = Vec<Rule>;
pub(crate) type FactoredSolution<'a> = (Lexicon<'a>, Rules, Rules, Rules, Rules);
// type Solution<'a> = (Vec<&'a Rule>, Rules);

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
    DeleteRules,
    Combine,
    Compose,
    ComposeDeep,
    RecurseDeep,
    GeneralizeDeep,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum TRSMove {
    Memorize(bool),
    SampleRule((f64, f64, f64, f64), usize),
    RegenerateRule((f64, f64, f64, f64), usize),
    LocalDifference,
    MemorizeOne,
    DeleteRule,
    Variablize,
    Generalize,
    Recurse(usize),
    DeleteRules(usize),
    Combine(usize),
    Compose,
    ComposeDeep,
    RecurseDeep(usize),
    GeneralizeDeep,
}
impl TRSMove {
    #[allow(clippy::too_many_arguments)]
    pub fn take<'a, 'b, R: Rng>(
        &self,
        gp: &TRSGP<'a, 'b>,
        task: &Task<Lexicon<'b>, TRS<'a, 'b>, Vec<Rule>>,
        obs: &[Rule],
        rng: &mut R,
        parents: &[&TRS<'a, 'b>],
        params: &GeneticParamsFull,
        gpparams: &GPParams,
    ) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        match *self {
            TRSMove::Memorize(deterministic) => {
                Ok(TRS::memorize(&gp.lexicon, deterministic, &gp.bg, obs))
            }
            TRSMove::SampleRule(aw, mss) => parents[0].sample_rule(aw, mss, rng),
            TRSMove::RegenerateRule(aw, mss) => parents[0].regenerate_rule(aw, mss, rng),
            TRSMove::LocalDifference => parents[0].local_difference(rng),
            TRSMove::MemorizeOne => parents[0].memorize_one(obs),
            TRSMove::DeleteRule => parents[0].delete_rule(),
            TRSMove::Variablize => parents[0].variablize(),
            TRSMove::Generalize => parents[0].generalize(),
            TRSMove::Recurse(n) => parents[0].recurse(n),
            TRSMove::DeleteRules(t) => parents[0].delete_rules(rng, t),
            TRSMove::Combine(t) => TRS::combine(&parents[0], &parents[1], rng, t),
            TRSMove::Compose => parents[0].compose(),
            TRSMove::ComposeDeep => parents[0]
                .compose()
                .and_then(|trss| TRS::nest(&trss, task, gp, rng, params, gpparams)),
            TRSMove::RecurseDeep(n) => parents[0]
                .recurse(n)
                .and_then(|trss| TRS::nest(&trss, task, gp, rng, params, gpparams)),
            TRSMove::GeneralizeDeep => parents[0]
                .generalize()
                .and_then(|trss| TRS::nest(&trss, task, gp, rng, params, gpparams)),
        }
    }
    pub fn get_parents<'a, 'b, 'c, R: Rng>(
        &self,
        t: &'c Tournament<TRS<'a, 'b>>,
        rng: &mut R,
    ) -> Vec<&'c TRS<'a, 'b>> {
        match *self {
            TRSMove::Memorize(_) => vec![],
            TRSMove::Combine(_) => vec![t.sample(rng), t.sample(rng)],
            _ => vec![t.sample(rng)],
        }
    }
    pub(crate) fn name(&self) -> TRSMoveName {
        match *self {
            TRSMove::Memorize(_) => TRSMoveName::Memorize,
            TRSMove::SampleRule(..) => TRSMoveName::SampleRule,
            TRSMove::RegenerateRule(..) => TRSMoveName::RegenerateRule,
            TRSMove::LocalDifference => TRSMoveName::LocalDifference,
            TRSMove::MemorizeOne => TRSMoveName::MemorizeOne,
            TRSMove::DeleteRule => TRSMoveName::DeleteRule,
            TRSMove::Variablize => TRSMoveName::Variablize,
            TRSMove::Generalize => TRSMoveName::Generalize,
            TRSMove::Recurse(..) => TRSMoveName::Recurse,
            TRSMove::DeleteRules(..) => TRSMoveName::DeleteRules,
            TRSMove::Combine(..) => TRSMoveName::Combine,
            TRSMove::Compose => TRSMoveName::Compose,
            TRSMove::ComposeDeep => TRSMoveName::ComposeDeep,
            TRSMove::RecurseDeep(..) => TRSMoveName::RecurseDeep,
            TRSMove::GeneralizeDeep => TRSMoveName::GeneralizeDeep,
        }
    }
}

/// A typed term rewriting system.
#[derive(Debug, PartialEq, Clone)]
pub struct TRS<'a, 'b> {
    pub(crate) lex: Lexicon<'b>,
    // INVARIANT: utrs never contains background information
    pub(crate) background: &'a [Rule],
    pub(crate) bg_ops: HashMap<Operator, Operator>,
    pub(crate) bg_vars: HashMap<Variable, Variable>,
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
        let bg_ops: HashMap<Operator, Operator> = background
            .iter()
            .flat_map(Rule::operators)
            .unique()
            .map(|o| (o, o))
            .collect();
        let bg_vars: HashMap<Variable, Variable> = background
            .iter()
            .flat_map(Rule::variables)
            .unique()
            .map(|v| (v, v))
            .collect();
        TRS {
            lex,
            background,
            bg_ops,
            bg_vars,
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

    pub fn same_shape(trs1: &TRS, trs2: &TRS) -> bool {
        trs1.len() == trs2.len()
            && trs1.background == trs2.background
            && UntypedTRS::same_shape_given(
                &trs1.utrs,
                &trs2.utrs,
                &mut trs1.bg_ops.clone(),
                &mut trs1.bg_vars.clone(),
            )
    }

    pub fn unique_shape(&self, trss: &[TRS]) -> bool {
        !trss.iter().any(|other| TRS::same_shape(self, other))
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
    fn contains(&self, rule: &Rule) -> bool {
        self.utrs.get_clause(rule).is_some()
    }
    fn remove_clauses<R>(&mut self, rules: &[R]) -> Result<(), SampleError>
    where
        R: Borrow<Rule>,
    {
        for rule in rules {
            if self.contains(rule.borrow()) {
                self.utrs.remove_clauses(rule.borrow())?;
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
                "EMPTY/0: list;",
                "HEAD/0: list -> nat;",
                "TAIL/0: list -> list;",
                "ISEMPTY/0: list -> bool;",
                "ISEQUAL/0: t1. t1 -> t1 -> bool;",
                "IF/0: t1. bool -> t1 -> t1 -> t1;",
                "TRUE/0: bool;",
                "FALSE/0: bool;",
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
    pub fn is_alpha_test_0() {
        let mut lex = create_test_lexicon();
        let bg = vec![
            parse_rule("ISEMPTY EMPTY = TRUE", &mut lex).expect("parsed rule 1"),
            parse_rule("ISEMPTY (CONS x_ y_) = FALSE", &mut lex).expect("parsed rule 2"),
            parse_rule("ISEQUAL x_ x_ = TRUE", &mut lex).expect("parsed rule 3"),
            parse_rule("ISEQUAL x_ y_ = FALSE", &mut lex).expect("parsed rule 4"),
            parse_rule("HEAD (CONS x_ y_) = x_", &mut lex).expect("parsed rule 5"),
            parse_rule("IF TRUE  x_ y_ = x_", &mut lex).expect("parsed rule 6"),
            parse_rule("IF FALSE x_ y_ = y_", &mut lex).expect("parsed rule 7"),
            parse_rule("TAIL EMPTY = EMPTY", &mut lex).expect("parsed rule 8"),
            parse_rule("TAIL (CONS x_ y_) = y_", &mut lex).expect("parsed rule 9"),
        ];

        let mut lex1 = lex.clone();
        lex1.invent_operator(Some("F".to_string()), 0, &tp![@arrow[tp!(int), tp!(int)]]);
        let trs1 = parse_trs("F 0 = 1; F 1 = 0;", &mut lex1, true, &bg[..]).expect("parsed trs");

        let mut lex2 = lex.clone();
        lex2.invent_operator(Some("F".to_string()), 0, &tp![@arrow[tp!(int), tp!(int)]]);
        let trs2 = parse_trs("F 0 = 1; F 1 = 0;", &mut lex2, true, &bg[..]).expect("parsed trs");

        assert!(TRS::is_alpha(&trs1, &trs2));
        assert!(TRS::same_shape(&trs1, &trs2));
    }

    #[test]
    pub fn is_alpha_test_1() {
        let mut lex = create_test_lexicon();
        let bg = vec![
            parse_rule("ISEMPTY EMPTY = TRUE", &mut lex).expect("parsed rule 1"),
            parse_rule("ISEMPTY (CONS x_ y_) = FALSE", &mut lex).expect("parsed rule 2"),
            parse_rule("ISEQUAL x_ x_ = TRUE", &mut lex).expect("parsed rule 3"),
            parse_rule("ISEQUAL x_ y_ = FALSE", &mut lex).expect("parsed rule 4"),
            parse_rule("HEAD (CONS x_ y_) = x_", &mut lex).expect("parsed rule 5"),
            parse_rule("IF TRUE  x_ y_ = x_", &mut lex).expect("parsed rule 6"),
            parse_rule("IF FALSE x_ y_ = y_", &mut lex).expect("parsed rule 7"),
            parse_rule("TAIL EMPTY = EMPTY", &mut lex).expect("parsed rule 8"),
            parse_rule("TAIL (CONS x_ y_) = y_", &mut lex).expect("parsed rule 9"),
        ];

        let mut lex1 = lex.clone();
        lex1.invent_operator(Some("F".to_string()), 0, &tp![@arrow[tp!(int), tp!(int)]]);
        let trs1 = parse_trs("F 0 = 1; F 1 = 0;", &mut lex1, true, &bg[..]).expect("parsed trs");

        let mut lex2 = lex.clone();
        lex2.invent_operator(Some("F".to_string()), 0, &tp![@arrow[tp!(bool), tp!(int)]]);
        lex2.invent_operator(Some("G".to_string()), 0, &tp![@arrow[tp!(int), tp!(int)]]);
        let trs2 = parse_trs("G 0 = 1; G 1 = 0;", &mut lex2, true, &bg[..]).expect("parsed trs");

        assert!(!TRS::is_alpha(&trs1, &trs2));
        assert!(TRS::same_shape(&trs1, &trs2));
    }

    #[test]
    pub fn is_alpha_test_2() {
        let mut lex = create_test_lexicon();
        let bg = vec![
            parse_rule("ISEMPTY EMPTY = TRUE", &mut lex).expect("parsed rule 1"),
            parse_rule("ISEMPTY (CONS x_ y_) = FALSE", &mut lex).expect("parsed rule 2"),
            parse_rule("ISEQUAL x_ x_ = TRUE", &mut lex).expect("parsed rule 3"),
            parse_rule("ISEQUAL x_ y_ = FALSE", &mut lex).expect("parsed rule 4"),
            parse_rule("HEAD (CONS x_ y_) = x_", &mut lex).expect("parsed rule 5"),
            parse_rule("IF TRUE  x_ y_ = x_", &mut lex).expect("parsed rule 6"),
            parse_rule("IF FALSE x_ y_ = y_", &mut lex).expect("parsed rule 7"),
            parse_rule("TAIL EMPTY = EMPTY", &mut lex).expect("parsed rule 8"),
            parse_rule("TAIL (CONS x_ y_) = y_", &mut lex).expect("parsed rule 9"),
        ];

        let mut lex1 = lex.clone();
        lex1.invent_operator(Some("F".to_string()), 0, &tp![@arrow[tp!(int), tp!(int)]]);
        let trs1 = parse_trs("F 0 = 1; F 1 = 0;", &mut lex1, true, &bg[..]).expect("parsed trs");

        let mut lex2 = lex.clone();
        lex2.invent_operator(Some("F".to_string()), 0, &tp![@arrow[tp!(int), tp!(int)]]);
        lex2.invent_operator(Some("G".to_string()), 0, &tp![@arrow[tp!(int), tp!(int)]]);
        let trs2 = parse_trs("G 0 = 1; F 1 = 0;", &mut lex2, true, &bg[..]).expect("parsed trs");

        assert!(!TRS::is_alpha(&trs1, &trs2));
        assert!(!TRS::same_shape(&trs1, &trs2));
    }
}

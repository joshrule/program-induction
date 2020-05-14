//! Polymorphically typed (Hindley-Milner) First-Order Term Rewriting Systems (no abstraction)
//!
//! Much thanks to:
//! - https://github.com/rob-smallshire/hindley-milner-python
//! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! - (TAPL; Pierce, 2002, ch. 22)

// TODO: reinstate
// mod compose;
// mod generalize;
// mod recurse;
mod combine;
mod delete_rule;
mod lgg;
mod local_difference;
mod log_likelihood;
mod log_posterior;
mod log_prior;
mod memorize;
mod move_rule;
mod regenerate_rule;
mod sample_rule;
mod swap_lhs_and_rhs;
mod variablize;

//pub use self::compose::Composition;
//pub use self::recurse::Recursion;
//pub use self::variablize::{Types, Variablization};
use itertools::Itertools;
use rand::{seq::IteratorRandom, Rng};
use std::{borrow::Borrow, collections::HashMap, fmt};
use term_rewriting::{MergeStrategy, Operator, Rule, Term, TRS as UntypedTRS};
use trs::{Lexicon, SampleError, TypeError};

pub(crate) type Rules = Vec<Rule>;
pub(crate) type FactoredSolution<'ctx, 'lex> = (Lexicon<'ctx, 'lex>, Rules, Rules, Rules, Rules);

/// A typed term rewriting system.
#[derive(Debug, PartialEq, Clone, Eq)]
pub struct TRS<'ctx, 'b> {
    pub(crate) lex: Lexicon<'ctx, 'b>,
    // INVARIANT: utrs never contains background information
    pub(crate) background: &'b [Rule],
    pub(crate) bg_ops: HashMap<Operator, Operator>,
    pub(crate) utrs: UntypedTRS,
}
impl<'ctx, 'b> TRS<'ctx, 'b> {
    /// Create a new `TRS` under the given [`Lexicon`]. Any background knowledge
    /// will be appended to the given ruleset.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::{parse_lexicon, parse_rule, TRS};
    /// # use polytype::atype::{with_ctx};
    /// with_ctx(32, |ctx| {
    ///     let mut lex = parse_lexicon(
    ///         "PLUS/2: int -> int -> int; SUCC/1: int-> int; ZERO/0: int;",
    ///         &ctx,
    ///     ).expect("lex");
    ///
    ///     let rules = vec![
    ///         parse_rule("PLUS(x_ ZERO) = x_", &mut lex).expect("rule 1"),
    ///         parse_rule("PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))", &mut lex).expect("rule 2"),
    ///     ];
    ///
    ///     let trs = TRS::new(&lex, true, &[], rules).unwrap();
    ///
    ///     assert_eq!(trs.len(), 2);
    /// })
    /// ```
    ///
    /// [`Lexicon`]: struct.Lexicon.html
    pub fn new(
        lexicon: &Lexicon<'ctx, 'b>,
        deterministic: bool,
        background: &'b [Rule],
        rules: Vec<Rule>,
    ) -> Result<Self, TypeError<'ctx>> {
        let trs = TRS::new_unchecked(lexicon, deterministic, background, rules);
        lexicon.infer_utrs(&trs.utrs)?;
        Ok(trs)
    }

    /// Like [`TRS::new`] but skips type inference. This is useful in scenarios
    /// where you are already confident in the type safety of the new rules.
    ///
    /// [`TRS::new`]: struct.TRS.html#method.new
    pub fn new_unchecked(
        lexicon: &Lexicon<'ctx, 'b>,
        deterministic: bool,
        background: &'b [Rule],
        rules: Vec<Rule>,
    ) -> Self {
        // Remove any rules already in the background
        let mut utrs = UntypedTRS::new(rules);
        for bg in background.iter().flat_map(|r| r.clauses()) {
            utrs.remove_clauses(&bg).ok();
        }
        if deterministic {
            utrs.make_deterministic();
        }
        //let lex = lexicon.clone();
        let bg_ops: HashMap<Operator, Operator> = HashMap::new();
        TRS {
            lex: lexicon.clone(),
            background,
            bg_ops,
            utrs,
        }
    }

    pub fn identify_symbols(&mut self) {
        self.bg_ops = self
            .lex
            .signature()
            .operators()
            .iter()
            .map(|o| (*o, *o))
            .collect()
    }

    pub fn lexicon(&self) -> &Lexicon<'ctx, 'b> {
        &self.lex
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
    ) -> Result<&mut Self, SampleError<'ctx>> {
        self.utrs.replace(n, old_clause, new_clause)?;
        Ok(self)
    }

    pub fn swap_rules(&mut self, rules: &[(Rule, Rule)]) -> Result<&mut Self, SampleError<'ctx>> {
        for (new_rule, old_rule) in rules {
            self.swap(&old_rule, new_rule.clone())?;
        }
        Ok(self)
    }

    pub fn swap(
        &mut self,
        old_rule: &Rule,
        new_rule: Rule,
    ) -> Result<&mut Self, SampleError<'ctx>> {
        if let Some((n, _)) = self.utrs.get_clause(old_rule) {
            self.utrs.replace(n, old_rule, new_rule)?;
        } else {
            self.utrs.insert(0, new_rule)?;
        }
        Ok(self)
    }

    pub fn is_alpha(trs1: &Self, trs2: &Self) -> bool {
        if trs1.len() != trs2.len() || trs1.background != trs2.background {
            return false;
        }
        if let Ok((lex, sig_change)) =
            Lexicon::merge(&trs1.lex, &trs2.lex, MergeStrategy::OperatorsByArityAndName)
        {
            let reified_trs2 = sig_change.reify_trs(&lex.lex.sig, trs2.utrs());
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
                &mut HashMap::new(),
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
    fn choose_clause<R: Rng>(&self, rng: &mut R) -> Result<(usize, Rule), SampleError<'ctx>> {
        let mut clauses = self.clauses();
        let idx = (0..clauses.len())
            .choose(rng)
            .ok_or(SampleError::OptionsExhausted)?;
        Ok(clauses.swap_remove(idx))
    }
    fn contains(&self, rule: &Rule) -> bool {
        self.utrs.get_clause(rule).is_some()
    }
    fn remove_clauses<R>(&mut self, rules: &[R]) -> Result<(), SampleError<'ctx>>
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
    fn prepend_clauses(&mut self, rules: Vec<Rule>) -> Result<(), SampleError<'ctx>> {
        self.utrs
            .pushes(rules)
            .map(|_| ())
            .map_err(SampleError::from)
    }
    pub(crate) fn append_clauses(&mut self, rules: Vec<Rule>) -> Result<(), SampleError<'ctx>> {
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
                    .all(|bg| Term::alpha(&[(&bg.lhs, &rule.lhs)]).is_none())
            });
        }
        rules.retain(|rule| !rule.is_empty());
    }
    /// Is the `Lexicon` deterministic?
    pub fn is_deterministic(&self) -> bool {
        self.utrs.is_deterministic()
    }
    /// Replace the current rules with a new set.
    pub fn adopt_rules(&self, rules: &mut Vec<Rule>) -> Option<Self> {
        self.filter_background(rules);

        let mut i = 0;
        while i < rules.len() {
            if rules[..i]
                .iter()
                .any(|other| Rule::alpha(&other, &rules[i]).is_some())
            {
                rules.remove(i);
            } else if self.is_deterministic()
                && rules[..i]
                    .iter()
                    .any(|other| Term::alpha(&[(&other.lhs, &rules[i].lhs)]).is_some())
            {
                return None;
            } else {
                i += 1;
            }
        }

        // Create a new TRS.
        let mut trs = self.clone();
        trs.utrs.rules = rules.to_vec();
        trs.smart_delete(0, 0).ok()
    }
}
impl<'ctx, 'b> fmt::Display for TRS<'ctx, 'b> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let trs_str = self
            .utrs
            .rules
            .iter()
            .map(|r| format!("{};", r.display(&self.lex.lex.sig)))
            .join("\n");

        write!(f, "{}", trs_str)
    }
}

#[cfg(test)]
mod tests {
    use polytype::atype::{with_ctx, TypeContext};
    use trs::parser::{parse_lexicon, parse_rule, parse_trs};
    use trs::{Lexicon, TRS};

    fn create_test_lexicon<'ctx, 'b>(ctx: &TypeContext<'b>) -> Lexicon<'b, 'ctx> {
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
            &ctx,
        )
        .expect("parsed lexicon")
    }

    #[test]
    pub fn is_alpha_test_0() {
        with_ctx(32, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
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
            let t_int = ctx.intern_tcon(ctx.intern_name("int"), &[]);
            let tp = ctx.arrow(t_int, t_int);
            lex1.invent_operator(Some("F".to_string()), 0, tp);
            let trs1 =
                parse_trs("F 0 = 1; F 1 = 0;", &mut lex1, true, &bg[..]).expect("parsed trs");

            let mut lex2 = lex.clone();
            lex2.invent_operator(Some("F".to_string()), 0, tp);
            let trs2 =
                parse_trs("F 0 = 1; F 1 = 0;", &mut lex2, true, &bg[..]).expect("parsed trs");

            assert!(TRS::is_alpha(&trs1, &trs2));
            assert!(TRS::same_shape(&trs1, &trs2));
        })
    }

    #[test]
    pub fn is_alpha_test_1() {
        with_ctx(32, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
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
            let t_int = ctx.intern_tcon(ctx.intern_name("int"), &[]);
            let t_bool = ctx.intern_tcon(ctx.intern_name("bool"), &[]);
            let tp = ctx.arrow(t_int, t_int);
            let distractor = ctx.arrow(t_bool, t_int);
            lex1.invent_operator(Some("F".to_string()), 0, tp);
            let trs1 =
                parse_trs("F 0 = 1; F 1 = 0;", &mut lex1, true, &bg[..]).expect("parsed trs");

            let mut lex2 = lex.clone();
            lex2.invent_operator(Some("F".to_string()), 0, distractor);
            lex2.invent_operator(Some("G".to_string()), 0, tp);
            let trs2 =
                parse_trs("G 0 = 1; G 1 = 0;", &mut lex2, true, &bg[..]).expect("parsed trs");

            assert!(!TRS::is_alpha(&trs1, &trs2));
            assert!(TRS::same_shape(&trs1, &trs2));
        })
    }

    #[test]
    pub fn is_alpha_test_2() {
        with_ctx(32, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
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
            let t_int = ctx.intern_tcon(ctx.intern_name("int"), &[]);
            let tp = ctx.arrow(t_int, t_int);
            lex1.invent_operator(Some("F".to_string()), 0, tp);
            let trs1 =
                parse_trs("F 0 = 1; F 1 = 0;", &mut lex1, true, &bg[..]).expect("parsed trs");

            let mut lex2 = lex.clone();
            lex2.invent_operator(Some("F".to_string()), 0, tp);
            lex2.invent_operator(Some("G".to_string()), 0, tp);
            let trs2 =
                parse_trs("G 0 = 1; F 1 = 0;", &mut lex2, true, &bg[..]).expect("parsed trs");

            assert!(!TRS::is_alpha(&trs1, &trs2));
            assert!(!TRS::same_shape(&trs1, &trs2));
        })
    }

    #[test]
    pub fn same_shape_test() {
        with_ctx(32, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
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
            let trs1 = parse_trs("C = C;", &mut lex, true, &bg[..]).expect("parsed trs");
            let trs2 = parse_trs(".(C .(.(CONS .(DIGIT 9)) .(.(CONS .(DIGIT 5)) .(.(CONS .(DIGIT 9)) EMPTY)))) = .(.(CONS .(DIGIT 9)) EMPTY);", &mut lex, true, &bg[..]).expect("parsed trs");
            assert!(!TRS::same_shape(&trs1, &trs2));
            let trs1 = parse_trs("4 = 5;", &mut lex, true, &bg[..]).expect("parsed trs");
            let trs2 = parse_trs("5 = 4;", &mut lex, true, &bg[..]).expect("parsed trs");
            assert!(TRS::same_shape(&trs1, &trs2));
        })
    }
}

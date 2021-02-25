use crate::{
    hypotheses::{BayesScore, Bayesable, Created, Hypothesis, MCMCable, Temperable},
    inference::Proposal,
    trs::{Composition, Datum, Env, ModelParams, Recursion, SingleLikelihood, Variablization, TRS},
    tryo,
};
use itertools::Itertools;
use polytype::atype::Ty;
use rand::{distributions::weighted::WeightedIndex, prelude::*};
use std::{collections::HashMap, convert::TryFrom, f64::NAN, fmt, fmt::Display, hash::Hash};
use term_rewriting::{Atom, Context, Rule, RuleContext, SituatedAtom, Term};
use utilities::f64_eq;

#[derive(Clone, Debug)]
pub struct MetaProgram<'ctx, 'b> {
    seed: TRS<'ctx, 'b>,
    moves: Vec<Vec<Move<'ctx>>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct State<'ctx, 'b> {
    pub(crate) data: &'b [&'b Datum],
    pub trs: TRS<'ctx, 'b>,
    pub n: usize,
    pub path: MetaProgram<'ctx, 'b>,
    pub(crate) spec: Option<MoveState<'ctx, 'b>>,
    pub(crate) label: StateLabel,
    pub probs: Vec<usize>,
}

#[derive(Clone)]
pub struct MetaProgramHypothesis<'ctx, 'b> {
    pub ctl: &'b MetaProgramControl<'b>,
    pub birth: BirthRecord,
    pub state: State<'ctx, 'b>,
    pub ln_meta: f64,
    pub ln_trs: f64,
    pub ln_acc: f64,
    pub ln_wf: f64,
    pub score: BayesScore,
}

pub struct MetaProgramControl<'b> {
    pub data: &'b [&'b Datum],
    pub model: &'b ModelParams,
    pub max_revisions: usize,
    pub max_length: usize,
    pub trs_temperature: f64,
}

impl<'b> MetaProgramControl<'b> {
    pub fn new(
        data: &'b [&'b Datum],
        model: &'b ModelParams,
        max_revisions: usize,
        max_length: usize,
        trs_temperature: f64,
    ) -> Self {
        MetaProgramControl {
            data,
            model,
            max_revisions,
            max_length,
            trs_temperature,
        }
    }
}

pub struct MetaProgramIter<'a, 'ctx, 'b> {
    it: &'a MetaProgram<'ctx, 'b>,
    idx: usize,
    jdx: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct BirthRecord {
    pub time: f64,
    pub count: usize,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum MoveState<'ctx, 'b> {
    SampleRule(Box<RuleContext>, Box<Env<'ctx, 'b>>, Vec<Ty<'ctx>>),
    RegenerateRule(RegenerateRuleState<'ctx, 'b>),
    Compose,
    Recurse,
    Variablize,
    MemorizeDatum,
    DeleteRule,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum RegenerateRuleState<'ctx, 'b> {
    Start,
    Rule(usize),
    Place(usize, Vec<usize>),
    Term(usize, Box<RuleContext>, Box<Env<'ctx, 'b>>, Vec<Ty<'ctx>>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Move<'ctx> {
    SampleRule,
    SampleAtom(Option<Atom>),
    RegenerateRule,
    RegenerateThisRule(usize),
    RegenerateThisPlace(Option<usize>),
    MemorizeDatum(Option<usize>),
    DeleteRule(Option<usize>),
    Variablize(Option<Box<Variablization<'ctx>>>),
    Compose(Option<Box<Composition<'ctx>>>),
    Recurse(Option<Box<Recursion<'ctx>>>),
    MemorizeAll,
    Generalize,
    AntiUnify,
    Stop,
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum StateLabel {
    Failed,
    Terminal,
    PartialRevision,
    CompleteRevision,
}

/// Generate a list of `Atom`s that can fit into a `RuleContext`'s `Hole`.
///
/// # Examples
///
/// ```
/// # extern crate polytype;
/// # extern crate term_rewriting;
/// # extern crate programinduction;
/// # use polytype::atype::{with_ctx, TypeContext, Variable as TVar};
/// # use term_rewriting::{Atom, Context, SituatedAtom, Variable};
/// # use programinduction::trs::{Env, Lexicon, TRS, parse_rulecontext, parse_lexicon};
/// # use programinduction::trs::mcts::rulecontext_fillers;
/// with_ctx(1024, |ctx| {
///
///   let mut lex = parse_lexicon(
///       &[
///           "C/0: list -> list;",
///           "CONS/0: nat -> list -> list;",
///           "NIL/0: list;",
///           "DECC/0: nat -> int -> nat;",
///           "DIGIT/0: int -> nat;",
///           "+/0: nat -> nat -> nat;",
///           "-/0: nat -> nat -> nat;",
///           ">/0: nat -> nat -> bool;",
///           "TRUE/0: bool; FALSE/0: bool;",
///           "HEAD/0: list -> nat;",
///           "TAIL/0: list -> list;",
///           "EMPTY/0: list -> bool;",
///           "EQUAL/0: t0. t0 -> t0 -> bool;",
///           "IF/0: t0. bool -> t0 -> t0 -> t0;",
///           "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
///           "0/0: int; 1/0: int; 2/0: int;",
///           "3/0: int; 4/0: int; 5/0: int;",
///           "6/0: int; 7/0: int; 8/0: int;",
///           "9/0: int; NAN/0: nat",
///       ].join(" "),
///       &ctx,
///   ).expect("lex");
///   let context = parse_rulecontext("v0_ (> (HEAD [!])) = CONS", &mut lex).expect("context");
///   let mut env = lex.infer_rulecontext(&context).expect("env");
///   let tp = context
///                .preorder()
///                .zip(&env.tps)
///                .find(|(t, _)| t.is_hole())
///                .map(|(_, tp)| *tp)
///                .expect("tp");
///   let mut arg_tps = vec![tp];
///   let fillers = rulecontext_fillers(&context, &mut env, &mut arg_tps);
///   assert_eq!(fillers.len(), 3);
///   assert_eq!(
///       vec!["Some(\"NIL\")", "Some(\".\")", "None"],
///       fillers.iter().map(|a| format!("{:?}", a.map(|atom| atom.display(lex.signature())))).collect::<Vec<_>>()
///   );
/// })
/// ```
pub fn rulecontext_fillers<'ctx, 'b>(
    context: &RuleContext,
    env: &mut Env<'ctx, 'b>,
    arg_types: &mut Vec<Ty<'ctx>>,
) -> Vec<Option<Atom>> {
    match context.leftmost_hole() {
        None => vec![],
        Some(place) => {
            let lhs_hole = place[0] == 0;
            let full_lhs_hole = place == [0];
            env.invent = lhs_hole && !full_lhs_hole;
            env.enumerate_atoms(arg_types[0])
                .filter_map(|atom| match atom {
                    None if full_lhs_hole => None,
                    Some(a) if full_lhs_hole && a.is_variable() => None,
                    x => Some(x),
                })
                .collect_vec()
        }
    }
}

impl PartialEq for BirthRecord {
    fn eq(&self, other: &Self) -> bool {
        f64_eq(self.time, other.time) && self.count == other.count
    }
}

impl Eq for BirthRecord {}

impl<'ctx, 'b> PartialEq for MetaProgram<'ctx, 'b> {
    fn eq(&self, other: &Self) -> bool {
        self.seed == other.seed && self.moves == other.moves
    }
}

impl<'ctx, 'b> Eq for MetaProgram<'ctx, 'b> {}

impl<'ctx, 'b> From<TRS<'ctx, 'b>> for MetaProgram<'ctx, 'b> {
    fn from(trs: TRS<'ctx, 'b>) -> Self {
        MetaProgram::new(trs, vec![vec![Move::Stop]])
    }
}

impl<'a, 'ctx, 'b> Iterator for MetaProgramIter<'a, 'ctx, 'b> {
    type Item = &'a Move<'ctx>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.idx < self.it.moves.len() {
                if self.jdx < self.it.moves[self.idx].len() {
                    let item = &self.it.moves[self.idx][self.jdx];
                    self.jdx += 1;
                    return Some(item);
                } else {
                    self.jdx = 0;
                    self.idx += 1;
                }
            } else {
                return None;
            }
        }
    }
}

impl<'ctx, 'b> MetaProgram<'ctx, 'b> {
    pub fn new(seed: TRS<'ctx, 'b>, moves: Vec<Vec<Move<'ctx>>>) -> Self {
        MetaProgram { seed, moves }
    }
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }
    pub fn len(&self) -> usize {
        self.moves.len()
    }
    pub fn truncate(&mut self, idx: usize) {
        self.moves.truncate(idx);
    }
    pub fn add(&mut self, mv: Vec<Move<'ctx>>) {
        self.moves.push(mv);
    }
    // Note...I can cause all sorts of type problems.
    // TODO: refactor me to avoid failure.
    pub fn extend(&mut self, mv: Move<'ctx>) {
        if !self.moves.is_empty() {
            let idx = self.moves.len() - 1;
            self.moves[idx].push(mv);
        }
    }
    /// Iterate over steps in the meta-program, where each iteration yields a
    /// reference to the step and the number of alternatives.
    pub fn iter<'a>(&'a self) -> MetaProgramIter<'a, 'ctx, 'b> {
        MetaProgramIter {
            it: self,
            idx: 0,
            jdx: 0,
        }
    }
    pub fn regenerate_small<'g, R: Rng>(
        &'g self,
        ctl: &'g MetaProgramControl<'b>,
        rng: &'g mut R,
    ) -> Option<Proposal<Self>> {
        // Clone metaprogram.
        let mut meta = self.clone();

        // Decide what kind of move to make.
        // - Can only delete if the length is greater than 1.
        let greater_than_1 = (meta.len() > 1) as usize;
        let dist = WeightedIndex::new(&[1, greater_than_1, greater_than_1]).ok()?;
        match dist.sample(rng) {
            // Insert
            0 => {
                // Pick some index.
                let weights = (0..meta.len()).map(|_| 1.0).collect_vec();
                let dist = WeightedIndex::new(&weights).ok()?;
                let idx = dist.sample(rng);

                // Compute the old probabilities.
                let mut state = State::from_meta(meta, ctl);
                let old_p_meta = state.probs.iter().map(|x| -(*x as f64).ln()).sum::<f64>();
                let old_p_select = -(state.path.len() as f64).ln();

                // Insert a new move.
                // println!("#     idx: {}", idx);
                state = state.insert_move(ctl, idx, rng)?;
                meta = state.metaprogram()?;

                // Compute the new probabilities.
                let new_p_meta = state.probs.iter().map(|x| -(*x as f64).ln()).sum::<f64>();
                let new_p_select = -(state.path.len() as f64).ln();

                // Compute the FB probability.
                let fb = (-3f64.ln() + old_p_select + new_p_meta)
                    - (-3f64.ln() + new_p_select + old_p_meta);
                // println!("# INSERT old/new lengths: {}/{}", old_len, new_len);

                // Return.
                Some(Proposal(meta, fb))
            }
            // Delete
            1 => {
                // Pick some index.
                let weights = (0..(meta.len() - 1)).map(|_| 1.0).collect_vec();
                let dist = WeightedIndex::new(&weights).ok()?;
                let idx = dist.sample(rng);

                // Compute the old probabilities.
                let mut state = State::from_meta(meta, ctl);
                let old_p_meta = state.probs.iter().map(|x| -(*x as f64).ln()).sum::<f64>();
                let old_p_select = -(state.path.len() as f64).ln();

                // Delete that move.
                // println!("#     idx: {}", idx);
                meta = state.metaprogram()?;
                // println!("#     meta: {}", meta);
                meta.moves.remove(idx);
                // println!("#     meta after delete: {}", meta);
                state = State::from_meta(meta, ctl);
                meta = state.metaprogram()?;
                // println!("#     final meta: {}", meta);

                // Compute the new probabilities.
                let new_p_meta = state.probs.iter().map(|x| -(*x as f64).ln()).sum::<f64>();
                let new_p_select = -(state.path.len() as f64).ln();

                // Compute the FB probability.
                let fb = (-3f64.ln() + old_p_select + new_p_meta)
                    - (-3f64.ln() + new_p_select + old_p_meta);
                // println!("# DELETE old/new lengths: {}/{}", old_len, new_len);

                // Return.
                Some(Proposal(meta, fb))
            }
            // Replace
            2 => {
                // Pick some index (cannot pick Stop; it cannot change).
                let weights = (0..(meta.len() - 1)).map(|_| 1.0).collect_vec();
                let dist = WeightedIndex::new(&weights).ok()?;
                let oidx = dist.sample(rng);
                // println!("#     oidx: {}", oidx);

                // Pick some subindex.
                let move_len = meta.moves[oidx].len();
                let iidx = rng.gen_range(0..move_len);
                // println!("#     iidx: {}", iidx);

                // Compute the old probabilities.
                let mut state = State::from_meta(meta, ctl);
                let old_p_meta = state.probs.iter().map(|x| -(*x as f64).ln()).sum::<f64>();
                let old_p_oselect = -(state.path.len() as f64).ln();
                let old_p_iselect = -(move_len as f64).ln();

                // Regenerate the rest of the move from the point forward.
                state = state.replace_move(ctl, oidx, iidx, rng)?;
                meta = state.metaprogram()?;

                // Compute the new probabilities.
                let new_p_meta = state.probs.iter().map(|x| -(*x as f64).ln()).sum::<f64>();
                let move_len = meta.moves[oidx].len();
                let new_p_iselect = -(move_len as f64).ln();

                // Compute the FB probability.
                let fb = (-3f64.ln() + old_p_oselect + old_p_iselect + new_p_meta)
                    - (-3f64.ln() + old_p_oselect + new_p_iselect + old_p_meta);
                // println!("# REPLACE old/new lengths: {}/{}", old_len, new_len);

                // Return.
                Some(Proposal(meta, fb))
            }
            _ => unreachable!(),
        }
    }
    /// Regenerate a random subnode of `from`, and return it along with the
    /// forward-backward probability.
    pub fn regenerate<'g, R: Rng>(
        &'g self,
        ctl: &'g MetaProgramControl<'b>,
        rng: &'g mut R,
    ) -> Option<Proposal<Self>> {
        // Clone.
        let mut meta = self.clone();

        // JSR: Removed check for fixed subparts.

        // Pick some point in the meta-program.
        // // geometric from first
        // let weights = (0..meta.len())
        //     .map(|x| 0.25f64.powi((x as i32) + 1))
        //     .collect_vec();
        let weights = (0..meta.len()).map(|_| 1.0).collect_vec();
        let dist = WeightedIndex::new(&weights).ok()?;
        let idx = dist.sample(rng);

        // Compute the probability of what we had there.
        let mut state = State::from_meta(meta, ctl);
        let old_p_meta = state.probs.iter().map(|x| -(*x as f64).ln()).sum::<f64>();
        let old_p_select = -(state.path.len() as f64).ln();
        let old_len = state.path.len();

        // Truncate it.
        meta = state.metaprogram()?;
        meta.truncate(idx);
        // println!("meta: {}", meta);

        // Use the playout logic to complete the meta-program.
        state = State::from_meta(meta, ctl).playout(ctl, rng)?;
        meta = state.metaprogram()?;
        // println!("meta: {}", meta);

        // Compute the probability of what we have now.
        let new_p_meta = state.probs.iter().map(|x| -(*x as f64).ln()).sum::<f64>();
        let new_p_select = -(state.path.len() as f64).ln();
        let new_len = state.path.len();

        // Compute the overall FB probability.
        let fb = old_p_select + new_p_meta - (new_p_select + old_p_meta);

        // Return the result
        Some(Proposal(meta, fb))
    }
}

impl<'ctx, 'b> Display for MetaProgram<'ctx, 'b> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.moves.iter().format_with(", ", |e, f| {
                f(&format_args!("[{}]", e.iter().format(", ")))
            })
        )
    }
}

impl<'ctx, 'b> MetaProgramHypothesis<'ctx, 'b> {
    pub fn new(ctl: &'b MetaProgramControl<'b>, program: MetaProgram<'ctx, 'b>) -> Self {
        let birth = BirthRecord {
            time: 0.0,
            count: 0,
        };
        let mut h = MetaProgramHypothesis {
            // mcts,
            ctl,
            birth,
            state: State::from_meta(program, ctl),
            ln_meta: NAN,
            ln_trs: NAN,
            ln_acc: NAN,
            ln_wf: NAN,
            score: BayesScore::default(),
        };
        h.compute_posterior(ctl.data, None);
        h
    }
    // pub fn play(&self) -> Option<State<'ctx, 'b>> {
    //     // Create default state.
    //     let mut state = State::seed(self.mcts);
    //     // Make each move, failing if necessary.
    //     for mv in self.state.path.iter() {
    //         // Providing bogus count, since we don't care.
    //         state.make_move(mv, 1, self.mcts.data);
    //         if StateLabel::Failed == state.label {
    //             return None;
    //         }
    //     }
    //     Some(state)
    // }
}

impl<'ctx, 'b> Display for MetaProgramHypothesis<'ctx, 'b> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.state.path)
    }
}

impl<'ctx, 'b> PartialEq for MetaProgramHypothesis<'ctx, 'b> {
    fn eq(&self, other: &Self) -> bool {
        self.state == other.state
            && self.birth == other.birth
            && self.score == other.score
            && f64_eq(self.ln_acc, other.ln_acc)
            && f64_eq(self.ln_meta, other.ln_meta)
            && f64_eq(self.ln_trs, other.ln_trs)
            && f64_eq(self.ln_wf, other.ln_wf)
    }
}

impl<'ctx, 'b> Eq for MetaProgramHypothesis<'ctx, 'b> {}

impl<'ctx, 'b> Created for MetaProgramHypothesis<'ctx, 'b> {
    type Record = BirthRecord;
    fn created(&self) -> Self::Record {
        self.birth
    }
}

impl<'ctx, 'b> Hypothesis for MetaProgramHypothesis<'ctx, 'b> {}

impl<'ctx, 'b> Bayesable for MetaProgramHypothesis<'ctx, 'b> {
    type Datum = &'b Datum;
    fn bayes_score(&self) -> &BayesScore {
        &self.score
    }
    fn bayes_score_mut(&mut self) -> &mut BayesScore {
        &mut self.score
    }
    fn compute_prior(&mut self) -> f64 {
        // make a state
        self.ln_meta = self
            .state
            .probs
            .iter()
            .fold(0.0, |partial_prior, n| partial_prior - (*n as f64).ln());
        self.state.trs.utrs.canonicalize(&mut HashMap::new());
        self.ln_trs = self.state.trs.log_prior(self.ctl.model.prior);
        self.score.prior = self.ln_meta + self.ln_trs / self.ctl.trs_temperature;
        self.score.prior
    }
    fn compute_single_likelihood(&mut self, datum: &Self::Datum) -> f64 {
        self.state.trs.utrs.canonicalize(&mut HashMap::new());
        self.state
            .trs
            .single_log_likelihood(datum, self.ctl.model.likelihood)
            .unwrap_or(std::f64::NAN)
    }
    fn compute_likelihood(&mut self, data: &[Self::Datum], _breakout: Option<f64>) -> f64 {
        // TODO: Add breakout logic.
        self.state.trs.utrs.canonicalize(&mut HashMap::new());
        // TODO: this is awkward and hacky.
        let mut l2 = self.ctl.model.likelihood;
        l2.single = SingleLikelihood::Generalization(0.000001);
        self.ln_wf = self.state.trs.log_likelihood(data, l2);
        self.ln_acc = self
            .state
            .trs
            .log_likelihood(data, self.ctl.model.likelihood);
        self.score.likelihood = self.ln_acc + self.ln_wf;
        self.score.likelihood
    }
}

impl<'ctx, 'b> Temperable for MetaProgramHypothesis<'ctx, 'b> {
    fn at_temperature(&self, t: f64) -> f64 {
        let score = self.bayes_score();
        score.prior + score.likelihood / t
    }
}

impl<'ctx, 'b> MCMCable for MetaProgramHypothesis<'ctx, 'b> {
    fn restart<R: Rng>(&mut self, rng: &mut R) -> Self {
        loop {
            let meta = MetaProgram::from(self.state.path.seed.clone());
            match meta.regenerate(self.ctl, rng) {
                None => continue,
                Some(Proposal(meta, _)) => {
                    // Don't compute posterior here.
                    // TODO: fix incorrect birthtime.
                    return MetaProgramHypothesis::new(self.ctl, meta);
                }
            }
        }
    }
    fn propose<R: Rng>(&mut self, rng: &mut R) -> (Self, f64) {
        loop {
            match self.state.path.regenerate_small(self.ctl, rng) {
                None => continue,
                Some(Proposal(meta, fb)) => {
                    let mut h = self.clone();
                    // TODO: fix incorrect birthtime.
                    // Don't compute posterior here.
                    h.state = State::from_meta(meta, self.ctl);
                    return (h, fb);
                }
            }
        }
    }
    fn replicate(&mut self, other: &Self) {
        self.score = other.score;
        self.ln_meta = other.ln_meta;
        self.ln_acc = other.ln_acc;
        self.ln_wf = other.ln_wf;
        self.ln_trs = other.ln_trs;
        self.birth = other.birth;
    }
}

impl<'ctx, 'b> State<'ctx, 'b> {
    pub fn metaprogram(&self) -> Option<MetaProgram<'ctx, 'b>> {
        if StateLabel::Failed == self.label {
            None
        } else {
            Some(self.path.clone())
        }
    }
    pub fn trs(&self) -> Option<&TRS<'ctx, 'b>> {
        if StateLabel::Failed == self.label {
            None
        } else {
            Some(&self.trs)
        }
    }
    pub fn from_meta(path: MetaProgram<'ctx, 'b>, ctl: &MetaProgramControl<'b>) -> Self {
        let mut state = State {
            probs: Vec::with_capacity(path.len()),
            trs: path.seed.clone(),
            path: MetaProgram {
                seed: path.seed.clone(),
                moves: Vec::with_capacity(path.len()),
            },
            data: ctl.data,
            spec: None,
            n: 0,
            label: StateLabel::CompleteRevision,
        };
        state.extend_with(ctl, &path.moves);
        state
    }
    // pub fn seed(mcts: &TRSMCTS<'ctx, 'b>) -> Self {
    //     let mut trs = TRS::new_unchecked(&mcts.lexicon, mcts.deterministic, mcts.bg, vec![]);
    //     trs.utrs.lo = mcts.lo;
    //     trs.utrs.hi = mcts.hi;
    //     trs.identify_symbols();
    //     State::from_meta(MetaProgram::from(trs), mcts)
    // }
    fn insert_move<'g, R: Rng>(
        &self,
        ctl: &'g MetaProgramControl<'b>,
        idx: usize,
        rng: &'g mut R,
    ) -> Option<Self> {
        // 1) run self.program up to idx
        let mut meta = self.metaprogram()?;
        let remaining_path = meta.moves.split_off(idx);
        // println!("#     prefix: {}", meta);
        let mut state = State::from_meta(meta, ctl);
        // 2) insert a single new move at idx
        state = state.add_move(ctl, rng)?;
        // println!("#     with new move: {}", state.metaprogram()?);
        // 3) append rest of self.program
        // println!("#     remaining path: {:?}", remaining_path);
        state.extend_with(ctl, &remaining_path);
        // println!("#    after extending: {}", state.metaprogram()?);
        Some(state)
    }
    fn replace_move<'g, R: Rng>(
        &self,
        ctl: &'g MetaProgramControl<'b>,
        oidx: usize,
        iidx: usize,
        rng: &'g mut R,
    ) -> Option<Self> {
        // 1) Split the program and delete the part to be replaced.
        let mut meta = self.metaprogram()?;
        let remaining_path = meta.moves.split_off(oidx + 1);
        // println!("#     prefix: {}", meta);
        if iidx > 0 {
            meta.moves[oidx].truncate(iidx);
        } else {
            meta.moves.pop();
        }
        // println!("#     truncated: {}", meta);
        // 2) run self.program up to this point.
        let mut state = State::from_meta(meta, ctl);
        // 3) insert a single new move at idx
        state = state.finish_move(ctl, false, rng)?;
        // println!("#     with new move: {}", state.metaprogram()?);
        // println!("#     remaining path: {:?}", remaining_path);
        // 4) append rest of self.program
        state.extend_with(ctl, &remaining_path);
        // println!("#    after extending: {}", state.metaprogram()?);
        Some(state)
    }
    pub fn add_step<R: Rng>(
        mut self,
        ctl: &MetaProgramControl<'b>,
        stop: bool,
        rng: &mut R,
    ) -> Option<Self> {
        if self.path.len() >= ctl.max_length {
            return None;
        }
        // Compute the available moves.
        let moves = self.available_moves(ctl);
        // Choose a move (random policy favoring STOP).
        let moves_len = moves.len();
        let mut move_weights = std::iter::repeat(1.0).take(moves_len).collect_vec();
        // Choose `Stop` 1/3 the time.
        if let Some(idx) = moves.iter().position(|mv| *mv == Move::Stop) {
            if stop {
                move_weights[idx] = (moves_len - 1) as f64 / 3.0;
            } else {
                move_weights[idx] = 0.0;
            }
        }
        let mut dist = WeightedIndex::new(&move_weights).ok()?;
        let i = 0;
        let mut state = self.clone();
        while i < moves_len {
            let idx = dist.sample(rng);
            let mv = &moves[idx];
            state.make_move(&mv, moves_len, ctl.data);
            match state.label {
                StateLabel::Failed => {
                    dist.update_weights(&[(idx, &0.0)]).ok()?;
                }
                _ => return Some(state),
            }
        }
        None
    }
    pub fn playout<R: Rng>(&self, ctl: &MetaProgramControl<'b>, rng: &mut R) -> Option<Self> {
        // TODO: Maybe try backtracking instead of a fixed count?
        for _ in 0..10 {
            let mut state = self.clone();
            let mut length = state.path.len();
            while length < ctl.max_length {
                match state.add_step(ctl, true, rng) {
                    None => break,
                    Some(new_state) => match new_state.label {
                        StateLabel::Terminal => return Some(new_state),
                        StateLabel::Failed => unreachable!(),
                        _ => {
                            length += 1;
                            state = new_state;
                        }
                    },
                }
            }
        }
        None
    }
    pub fn finish_move<R: Rng>(
        &self,
        ctl: &MetaProgramControl<'b>,
        stop: bool,
        rng: &mut R,
    ) -> Option<Self> {
        // TODO: Maybe try backtracking instead of a fixed count?
        for _ in 0..10 {
            let mut state = self.clone();
            let mut length = state.path.len();
            while length < ctl.max_length {
                match state.add_step(ctl, stop, rng) {
                    None => break,
                    Some(new_state) => match new_state.label {
                        StateLabel::Terminal => return Some(new_state),
                        StateLabel::Failed => unreachable!(),
                        _ => {
                            if new_state.spec.is_none() {
                                return Some(new_state);
                            } else {
                                length += 1;
                                state = new_state;
                            }
                        }
                    },
                }
            }
        }
        None
    }
    pub fn add_move<R: Rng>(&self, ctl: &MetaProgramControl<'b>, rng: &mut R) -> Option<Self> {
        // Continue only if we are between moves.
        match self.spec {
            Some(_) => None,
            None => self.finish_move(ctl, false, rng),
        }
    }
    pub fn extend_with(&mut self, ctl: &MetaProgramControl<'b>, path: &[Vec<Move<'ctx>>]) {
        for mv in path.iter().flatten() {
            if StateLabel::Failed != self.label {
                let n = self.available_moves(ctl).len();
                self.make_move(mv, n, ctl.data);
            }
        }
    }
    pub fn available_moves(&mut self, ctl: &MetaProgramControl<'b>) -> Vec<Move<'ctx>> {
        let mut moves = vec![];
        match &mut self.spec {
            None => {
                // Search can always stop.
                moves.push(Move::Stop);
                if self.n < ctl.max_revisions {
                    // Search can always sample a new rule.
                    moves.push(Move::SampleRule);
                    // A TRS must have a rule in order to regenerate or generalize.
                    if !self.trs.is_empty() {
                        moves.push(Move::RegenerateRule);
                        moves.push(Move::Generalize);
                        moves.push(Move::Compose(None));
                        moves.push(Move::Recurse(None));
                        moves.push(Move::Variablize(None));
                    }
                    // A TRS must have >1 rule to delete without creating cycles.
                    // Anti-unification relies on having two rules to unify.
                    if self.trs.len() > 1 {
                        moves.push(Move::DeleteRule(None));
                        moves.push(Move::AntiUnify);
                    }
                    // We can only add data if there's data to add.
                    if ctl.data.iter().any(|datum| match datum {
                        Datum::Partial(_) => false,
                        Datum::Full(rule) => self.trs.utrs.get_clause(rule).is_none(),
                    }) {
                        moves.push(Move::MemorizeAll);
                        moves.push(Move::MemorizeDatum(None));
                    }
                }
            }
            Some(MoveState::Variablize) => self
                .trs
                .find_all_variablizations()
                .into_iter()
                .for_each(|v| moves.push(Move::Variablize(Some(Box::new(v))))),
            Some(MoveState::Compose) => self
                .trs
                .find_all_compositions()
                .into_iter()
                .for_each(|composition| moves.push(Move::Compose(Some(Box::new(composition))))),
            Some(MoveState::Recurse) => self
                .trs
                .find_all_recursions()
                .into_iter()
                .for_each(|recursion| moves.push(Move::Recurse(Some(Box::new(recursion))))),
            Some(MoveState::MemorizeDatum) => {
                (0..ctl.data.len())
                    .filter(|idx| match ctl.data[*idx] {
                        Datum::Partial(_) => false,
                        Datum::Full(ref rule) => self.trs.utrs.get_clause(rule).is_none(),
                    })
                    .for_each(|idx| moves.push(Move::MemorizeDatum(Some(idx))));
            }
            Some(MoveState::DeleteRule) => {
                (0..self.trs.len()).for_each(|idx| moves.push(Move::DeleteRule(Some(idx))));
            }
            Some(MoveState::SampleRule(ref context, ref mut env, ref mut arg_tps))
            | Some(MoveState::RegenerateRule(RegenerateRuleState::Term(
                _,
                ref context,
                ref mut env,
                ref mut arg_tps,
            ))) => {
                // TODO: make rulecontext_fillers an iterator.
                rulecontext_fillers(context, env, arg_tps)
                    .into_iter()
                    .map(Move::SampleAtom)
                    .for_each(|mv| moves.push(mv));
            }
            Some(MoveState::RegenerateRule(RegenerateRuleState::Start)) => {
                for i in 0..self.trs.utrs.rules.len() {
                    moves.push(Move::RegenerateThisRule(i));
                }
            }
            Some(MoveState::RegenerateRule(RegenerateRuleState::Rule(n))) => {
                for i in 0..=self.trs.utrs.rules[*n].rhs.len() {
                    moves.push(Move::RegenerateThisPlace(Some(i)));
                }
            }
            Some(MoveState::RegenerateRule(RegenerateRuleState::Place(n, place))) => {
                if let Some(term) = self.trs.utrs.rules[*n].at(&place) {
                    if let Term::Application { args, .. } = term {
                        for i in 0..args.len() {
                            moves.push(Move::RegenerateThisPlace(Some(i)));
                        }
                    }
                    moves.push(Move::RegenerateThisPlace(None));
                }
            }
        }
        moves
    }
    pub fn make_move(&mut self, mv: &Move<'ctx>, n: usize, data: &[&'b Datum]) {
        match *mv {
            Move::Stop => {
                self.n += 1;
                self.path.add(vec![mv.clone()]);
                self.probs.push(n);
                self.spec = None;
                self.label = StateLabel::Terminal;
            }
            Move::Generalize => {
                let trs = tryo![self, self.trs.generalize().ok()];
                if self.trs != trs {
                    self.trs = trs;
                    self.n += 1;
                    self.path.add(vec![mv.clone()]);
                    self.probs.push(n);
                    self.spec = None;
                    self.label = StateLabel::CompleteRevision;
                } else {
                    self.label = StateLabel::Failed;
                }
            }
            Move::AntiUnify => {
                let trs = tryo![self, self.trs.lgg().ok()];
                if self.trs != trs {
                    self.trs = trs;
                    self.n += 1;
                    self.path.add(vec![mv.clone()]);
                    self.probs.push(n);
                    self.spec = None;
                    self.label = StateLabel::CompleteRevision;
                } else {
                    self.label = StateLabel::Failed;
                }
            }
            Move::Compose(None) => {
                self.path.add(vec![mv.clone()]);
                self.probs.push(n);
                self.label = StateLabel::PartialRevision;
                self.spec.replace(MoveState::Compose);
            }
            Move::Compose(Some(ref composition)) => {
                let trs = tryo![self, self.trs.compose_by(composition)];
                if self.trs != trs {
                    self.trs = trs;
                    self.n += 1;
                    self.path.extend(mv.clone());
                    self.probs.push(n);
                    self.spec = None;
                    self.label = StateLabel::CompleteRevision;
                } else {
                    self.label = StateLabel::Failed;
                }
            }
            Move::Recurse(None) => {
                self.path.add(vec![mv.clone()]);
                self.probs.push(n);
                self.label = StateLabel::PartialRevision;
                self.spec.replace(MoveState::Recurse);
            }
            Move::Recurse(Some(ref recursion)) => {
                let trs = tryo![self, self.trs.recurse_by(recursion)];
                if self.trs != trs {
                    self.trs = trs;
                    self.n += 1;
                    self.path.extend(mv.clone());
                    self.probs.push(n);
                    self.spec = None;
                    self.label = StateLabel::CompleteRevision;
                } else {
                    self.label = StateLabel::Failed;
                }
            }
            Move::Variablize(None) => {
                self.path.add(vec![mv.clone()]);
                self.probs.push(n);
                self.label = StateLabel::PartialRevision;
                self.spec.replace(MoveState::Variablize);
            }
            Move::Variablize(Some(ref v)) => {
                let mut clauses = self.trs.utrs.clauses();
                if clauses.len() <= v.0 {
                    self.label = StateLabel::Failed;
                }
                clauses[v.0] = tryo![
                    self,
                    self.trs.apply_variablization(&v.1, &v.2, &clauses[v.0])
                ];
                // TODO: remove clone
                let trs = tryo![self, self.trs.clone().adopt_rules(&mut clauses)];
                if self.trs != trs {
                    self.trs = trs;
                    self.n += 1;
                    self.path.extend(mv.clone());
                    self.probs.push(n);
                    self.spec = None;
                    self.label = StateLabel::CompleteRevision;
                } else {
                    self.label = StateLabel::Failed;
                }
            }
            Move::DeleteRule(None) => {
                self.path.add(vec![mv.clone()]);
                self.probs.push(n);
                self.label = StateLabel::PartialRevision;
                self.spec.replace(MoveState::DeleteRule);
            }
            Move::DeleteRule(Some(r)) => {
                tryo![self, self.trs.utrs.remove_idx(r).ok()];
                self.n += 1;
                self.path.extend(mv.clone());
                self.probs.push(n);
                self.spec = None;
                self.label = StateLabel::CompleteRevision;
            }
            Move::MemorizeAll => {
                let new_data = data
                    .iter()
                    .filter_map(|d| match d {
                        Datum::Partial(_) => None,
                        Datum::Full(rule) => {
                            if self.trs.utrs.get_clause(rule).is_none() {
                                Some(rule.clone())
                            } else {
                                None
                            }
                        }
                    })
                    .collect_vec();
                if new_data.is_empty() {
                    self.label = StateLabel::Failed;
                } else {
                    tryo![self, self.trs.append_clauses(new_data).ok()];
                    self.n += 1;
                    self.path.add(vec![mv.clone()]);
                    self.probs.push(n);
                    self.spec = None;
                    self.label = StateLabel::CompleteRevision;
                }
            }
            Move::MemorizeDatum(None) => {
                self.path.add(vec![mv.clone()]);
                self.probs.push(n);
                self.label = StateLabel::PartialRevision;
                self.spec.replace(MoveState::MemorizeDatum);
            }
            Move::MemorizeDatum(Some(r)) => match data[r] {
                Datum::Partial(_) => panic!("can't memorize partial data"),
                Datum::Full(ref rule) => {
                    tryo![self, self.trs.append_clauses(vec![rule.clone()]).ok()];
                    self.n += 1;
                    self.path.extend(mv.clone());
                    self.probs.push(n);
                    self.spec = None;
                    self.label = StateLabel::CompleteRevision;
                }
            },
            Move::SampleRule => {
                self.path.add(vec![mv.clone()]);
                self.probs.push(n);
                self.label = StateLabel::PartialRevision;
                let context = RuleContext::default();
                let mut env = Env::new(true, &self.trs.lex, Some(self.trs.lex.lex.src));
                let tp = env.new_type_variable();
                let arg_tps = vec![tp, tp];
                self.spec.replace(MoveState::SampleRule(
                    Box::new(context),
                    Box::new(env),
                    arg_tps,
                ));
            }
            Move::RegenerateRule => {
                self.path.add(vec![mv.clone()]);
                self.probs.push(n);
                self.label = StateLabel::PartialRevision;
                self.spec
                    .replace(MoveState::RegenerateRule(RegenerateRuleState::Start));
            }
            Move::RegenerateThisRule(r) => {
                if self.trs.utrs.rules.len() <= r {
                    self.label = StateLabel::Failed;
                } else {
                    self.path.extend(mv.clone());
                    self.probs.push(n);
                    self.label = StateLabel::PartialRevision;
                    self.spec
                        .replace(MoveState::RegenerateRule(RegenerateRuleState::Rule(r)));
                }
            }
            Move::RegenerateThisPlace(p) => {
                self.path.extend(mv.clone());
                self.probs.push(n);
                self.label = StateLabel::PartialRevision;
                match self.spec.take() {
                    Some(MoveState::RegenerateRule(RegenerateRuleState::Rule(r))) => match p {
                        None => panic!("no place specified"),
                        Some(p) => {
                            let rule = &self.trs.utrs.rules[r];
                            let max_height = rule
                                .lhs
                                .height()
                                .max(rule.rhs.iter().map(|rhs| rhs.height()).max().unwrap_or(0));
                            let mut ps = Vec::with_capacity(max_height);
                            ps.push(p);
                            if rule.at(&ps).is_none() {
                                self.label = StateLabel::Failed;
                            } else {
                                self.spec.replace(MoveState::RegenerateRule(
                                    RegenerateRuleState::Place(r, ps),
                                ));
                            }
                        }
                    },
                    Some(MoveState::RegenerateRule(RegenerateRuleState::Place(r, mut ps))) => {
                        match p {
                            None => {
                                let context = RuleContext::from(&self.trs.utrs.rules[r]);
                                let context =
                                    context.replace(&ps, Context::Hole).expect("bad place");
                                let env =
                                    tryo![self, self.trs.lex.infer_rulecontext(&context).ok()];
                                let tp = context
                                    .preorder()
                                    .zip(&env.tps)
                                    .find(|(t, _)| t.is_hole())
                                    .map(|(_, tp)| *tp)
                                    .expect("tp");
                                let arg_tps = vec![tp];
                                self.spec.replace(MoveState::RegenerateRule(
                                    RegenerateRuleState::Term(
                                        r,
                                        Box::new(context),
                                        Box::new(env),
                                        arg_tps,
                                    ),
                                ));
                            }
                            Some(p) => {
                                ps.push(p);
                                let rule = &self.trs.utrs.rules[r];
                                if rule.at(&ps).is_none() {
                                    self.label = StateLabel::Failed;
                                } else {
                                    self.spec.replace(MoveState::RegenerateRule(
                                        RegenerateRuleState::Place(r, ps),
                                    ));
                                }
                            }
                        }
                    }
                    x => panic!("# MoveState doesn't match Move: {:?}", x),
                }
            }
            Move::SampleAtom(atom) => {
                let spec = self.spec.take();
                match spec {
                    Some(MoveState::SampleRule(context, mut env, arg_tps)) => {
                        let atom = atom.unwrap_or_else(|| {
                            env.new_variable().map(Atom::Variable).expect("variable")
                        });
                        let place = tryo![self, context.leftmost_hole()];
                        let new_context = tryo![
                            self,
                            context.replace(
                                &place,
                                Context::from(tryo![
                                    self,
                                    SituatedAtom::new(atom, self.trs.lex.signature())
                                ])
                            )
                        ];
                        match Rule::try_from(&new_context) {
                            Ok(rule) => {
                                tryo![self, self.trs.append_clauses(vec![rule]).ok()];
                                self.n += 1;
                                self.path.extend(mv.clone());
                                self.probs.push(n);
                                self.spec = None;
                                self.label = StateLabel::CompleteRevision;
                            }
                            Err(v) if v.is_empty() => {
                                self.label = StateLabel::Failed;
                            }
                            _ => {
                                if !env.contains(atom) {
                                    self.label = StateLabel::Failed;
                                    return;
                                }
                                let tp = arg_tps[0];
                                let mut new_arg_tps = tryo![self, env.check_atom(tp, atom).ok()];
                                new_arg_tps.extend_from_slice(&arg_tps[1..]);
                                self.path.extend(mv.clone());
                                self.probs.push(n);
                                self.label = StateLabel::PartialRevision;
                                self.spec.replace(MoveState::SampleRule(
                                    Box::new(new_context),
                                    env,
                                    new_arg_tps,
                                ));
                            }
                        }
                    }
                    Some(MoveState::RegenerateRule(RegenerateRuleState::Term(
                        r,
                        context,
                        mut env,
                        arg_tps,
                    ))) => {
                        let place = tryo![self, context.leftmost_hole()];
                        let lhs_hole = place[0] == 0;
                        let full_lhs_hole = place == [0];
                        env.invent = lhs_hole && !full_lhs_hole;
                        let atom = atom.unwrap_or_else(|| {
                            env.new_variable().map(Atom::Variable).expect("variable")
                        });
                        let subcontext = Context::from(tryo![
                            self,
                            SituatedAtom::new(atom, self.trs.lex.signature())
                        ]);
                        let new_context = tryo![self, context.replace(&place, subcontext)];
                        match Rule::try_from(&new_context) {
                            Ok(rule) => {
                                tryo![self, self.trs.utrs.remove_idx(r).ok()];
                                tryo![self, self.trs.utrs.insert_idx(r, rule).ok()];
                                self.n += 1;
                                self.path.extend(mv.clone());
                                self.probs.push(n);
                                self.spec = None;
                                self.label = StateLabel::CompleteRevision;
                            }
                            Err(v) if v.is_empty() => {
                                self.label = StateLabel::Failed;
                            }
                            _ => {
                                if !env.contains(atom) {
                                    self.label = StateLabel::Failed;
                                    return;
                                }
                                let tp = arg_tps[0];
                                let mut new_arg_tps = tryo![self, env.check_atom(tp, atom).ok()];
                                new_arg_tps.extend_from_slice(&arg_tps[1..]);
                                self.path.extend(mv.clone());
                                self.probs.push(n);
                                self.label = StateLabel::PartialRevision;
                                self.spec.replace(MoveState::RegenerateRule(
                                    RegenerateRuleState::Term(
                                        r,
                                        Box::new(new_context),
                                        env,
                                        new_arg_tps,
                                    ),
                                ));
                            }
                        }
                    }
                    ref x => panic!("* MoveState doesn't match Move: {:?}", x),
                }
            }
        }
    }
}

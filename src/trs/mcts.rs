use generational_arena::{Arena, Index};
use itertools::Itertools;
use mcts::{
    MoveEvaluator, MoveHandle, NodeHandle, NodeStatistic, SearchTree, State, StateEvaluator,
    TreeStore, MCTS,
};
use polytype::atype::Ty;
use rand::prelude::{Rng, SliceRandom};
//use serde_json::Value;
use std::{cmp::Ordering, collections::HashMap, convert::TryFrom, f64::NEG_INFINITY, hash::Hash};
use term_rewriting::{Atom, Context, Rule, RuleContext, SituatedAtom, Term};
use trs::{
    Composition, Datum, Env, Lexicon, ModelParams, Recursion, Schedule, SingleLikelihood,
    Variablization, TRS,
};
use utils::{logdiffexp, logsumexp};

macro_rules! r#tryo {
    ($state:ident, $expr:expr) => {
        match $expr {
            std::option::Option::Some(val) => val,
            std::option::Option::None => {
                $state.label = StateLabel::Failed;
                return;
            }
        }
    };
}

#[derive(Copy, Debug, PartialEq, Eq, Clone, Hash)]
pub enum MCTSState {
    Revision(RevisionHandle),
    Terminal(TerminalHandle),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct HypothesisHandle(Index);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct RevisionHandle(Index);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct TerminalHandle(Index);

#[derive(Debug, Clone)]
pub struct Revision {
    playout: PlayoutState<HypothesisHandle>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Terminal {
    trs: HypothesisHandle,
}

#[derive(Debug, Copy, Clone)]
pub enum PlayoutState<T: std::fmt::Debug + Copy> {
    Untried,
    Failed,
    Success(T),
}

pub struct BestSoFarMoveEvaluator;
pub struct RescaledByBestMoveEvaluator;
pub struct RelativeMassMoveEvaluator;
pub struct ThompsonMoveEvaluator;
pub struct MaxThompsonMoveEvaluator;
pub struct MCTSStateEvaluator;

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

pub struct TRSMCTS<'ctx, 'b> {
    pub lexicon: Lexicon<'ctx, 'b>,
    pub bg: &'b [Rule],
    pub deterministic: bool,
    pub lo: usize,
    pub hi: usize,
    pub data: &'b [&'b Datum],
    pub root: Option<MCTSState>,
    pub hypotheses: Arena<Box<MCTSObj<'ctx>>>,
    pub revisions: Arena<Revision>,
    pub terminals: Arena<Terminal>,
    pub model: ModelParams,
    pub params: MCTSParams,
    pub best: f64,
    pub search_time: f64,
    pub trial_start: Option<std::time::Instant>,
    pub count: usize,
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct MCTSParams {
    pub max_depth: usize,
    pub max_states: usize,
    pub max_revisions: usize,
    pub max_size: usize,
    pub atom_weights: (f64, f64, f64, f64),
    pub invent: bool,
    pub selection: Selection,
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct QN {
    pub q: f64,
    pub n: f64,
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct QNMean(QN);

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct QNN(QN);

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct QNMax(QN);

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct VNMax {
    pub scores: Vec<f64>,
    pub n: f64,
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum Selection {
    RelativeMassUCT,
    BestSoFarUCT,
    Thompson(u32),
    MaxThompson { schedule: Schedule, n_top: usize },
}

#[derive(Clone, PartialEq)]
pub struct MCTSObj<'ctx> {
    pub time: f64,
    pub count: usize,
    pub moves: Vec<Move<'ctx>>,
    pub obj_meta: f64,
    pub obj_trs: f64,
    pub obj_acc: f64,
    pub obj_gen: f64,
    pub ln_search_prior: f64,
    pub ln_search_likelihood: f64,
    pub ln_search_posterior: f64,
    pub ln_predict_prior: f64,
    pub ln_predict_likelihood: f64,
    pub ln_predict_posterior: f64,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TrueState<'ctx, 'b> {
    trs: TRS<'ctx, 'b>,
    n: usize,
    spec: Option<MoveState<'ctx, 'b>>,
    path: Vec<(Move<'ctx>, usize)>,
    label: StateLabel,
}

impl<'ctx> MCTSObj<'ctx> {
    pub fn new(
        time: f64,
        count: usize,
        moves: Vec<Move<'ctx>>,
        obj_meta: f64,
        obj_trs: f64,
        obj_acc: f64,
        obj_gen: f64,
        ln_search_prior: f64,
        ln_search_likelihood: f64,
        ln_search_posterior: f64,
        ln_predict_prior: f64,
        ln_predict_likelihood: f64,
        ln_predict_posterior: f64,
    ) -> Self {
        MCTSObj {
            time,
            count,
            moves,
            obj_meta,
            obj_trs,
            obj_acc,
            obj_gen,
            ln_search_prior,
            ln_search_likelihood,
            ln_search_posterior,
            ln_predict_prior,
            ln_predict_likelihood,
            ln_predict_posterior,
        }
    }
    pub fn play<'b>(&self, mcts: &TRSMCTS<'ctx, 'b>) -> Option<TRS<'ctx, 'b>> {
        // Create default state.
        let mut state = MCTSState::root_data(mcts);
        // Make each move, failing if necessary.
        for mv in &self.moves {
            // Providing bogus count, since we don't care.
            state.make_move(mv, 1, mcts.data);
            if StateLabel::Failed == state.label {
                return None;
            }
        }
        Some(state.trs)
    }
    pub fn test_path<'b>(&self, mcts: &TRSMCTS<'ctx, 'b>) -> bool {
        let mut state = MCTSState::root_data(mcts);
        let trial = mcts.data.len() - 1;
        for mv in &self.moves {
            if state
                .available_moves(mcts)
                .iter()
                .find(|available| *available == mv)
                .is_some()
            {
                // Providing bogus count, since we don't care.
                state.make_move(mv, 1, mcts.data);
                if state.label == StateLabel::Failed {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}

impl<'ctx> Eq for MCTSObj<'ctx> {}

impl<'ctx> std::ops::Index<HypothesisHandle> for Arena<Box<MCTSObj<'ctx>>> {
    type Output = MCTSObj<'ctx>;
    fn index(&self, index: HypothesisHandle) -> &Self::Output {
        &self[index.0]
    }
}

impl<'ctx> std::ops::IndexMut<HypothesisHandle> for Arena<Box<MCTSObj<'ctx>>> {
    fn index_mut(&mut self, index: HypothesisHandle) -> &mut Self::Output {
        &mut self[index.0]
    }
}

impl std::ops::Index<RevisionHandle> for Arena<Revision> {
    type Output = Revision;
    fn index(&self, index: RevisionHandle) -> &Self::Output {
        &self[index.0]
    }
}

impl std::ops::IndexMut<RevisionHandle> for Arena<Revision> {
    fn index_mut(&mut self, index: RevisionHandle) -> &mut Self::Output {
        &mut self[index.0]
    }
}

impl std::ops::Index<TerminalHandle> for Arena<Terminal> {
    type Output = Terminal;
    fn index(&self, index: TerminalHandle) -> &Self::Output {
        &self[index.0]
    }
}

impl std::ops::IndexMut<TerminalHandle> for Arena<Terminal> {
    fn index_mut(&mut self, index: TerminalHandle) -> &mut Self::Output {
        &mut self[index.0]
    }
}

pub fn relative_mass_uct(parent: &QN, child: &QN) -> f64 {
    let q = if child.q == std::f64::NEG_INFINITY && parent.q == std::f64::NEG_INFINITY {
        0.0
    } else {
        child.q - parent.q
    };
    q.exp() * parent.n / child.n + (parent.n.ln() / child.n).sqrt()
}

pub fn rescaled_by_best_uct(parent: &QN, child: &QN, best: f64) -> f64 {
    (child.q - best).exp() / child.n + (parent.n.ln() / child.n).sqrt()
}

pub fn best_so_far_uct(parent: &QN, child: &QN, mcts: &TRSMCTS) -> f64 {
    let best = mcts.best;
    let exploit = (child.q - best).exp();
    let explore = (parent.n.ln() / child.n).sqrt();
    exploit + explore
}

pub fn max_thompson_sample<R: Rng>(
    _parent: NodeHandle,
    child: NodeHandle,
    tree: &TreeStore<TRSMCTS>,
    mcts: &TRSMCTS,
    rng: &mut R,
) -> f64 {
    match mcts.params.selection {
        Selection::MaxThompson { schedule, .. } => {
            let node = tree.node(child);
            let temp = schedule.temperature(node.stats.n as f64);
            let raw_score = *node
                .stats
                .scores
                .choose_weighted(rng, |x| (x / temp).exp())
                .unwrap_or(&NEG_INFINITY);
            raw_score / temp
        }
        x => panic!("in max_thompson_sample but Selection is {:?}", x),
    }
}

//pub fn thompson_sample<R: Rng>(
//    parent: NodeHandle,
//    child: NodeHandle,
//    tree: &TreeStore<TRSMCTS>,
//    mcts: &TRSMCTS,
//    rng: &mut R,
//) -> f64 {
//    let mut source = {
//        let a = match mcts.params.selection {
//            Selection::Thompson(a) => a,
//            x => panic!("in thompson_sample but Selection is {:?}", x),
//        };
//        let b = tree.node(child).stats.0.n as u32;
//        if Bernoulli::from_ratio(b, a + b).unwrap().sample(rng) {
//            child
//        } else {
//            parent
//        }
//    };
//    let mut source_node = tree.node(source);
//    let mut n = rng.gen_range(0, source_node.stats.0.n as usize);
//    let mut outgoing = &source_node.outgoing;
//    let mut idx = 0;
//    while n != 0 {
//        match tree.mv(outgoing[idx]).child {
//            None => idx += 1,
//            Some(target) => {
//                let target_node = tree.node(target);
//                if n > target_node.stats.0.n as usize {
//                    idx += 1;
//                    n -= target_node.stats.0.n as usize;
//                } else {
//                    source = target;
//                    source_node = target_node;
//                    outgoing = &source_node.outgoing;
//                    idx = 0;
//                    n -= 1;
//                }
//            }
//        }
//    }
//    tree.node(source).evaluation
//}

pub fn compute_path<'ctx, 'b>(
    parent: NodeHandle,
    mv: &Move<'ctx>,
    tree: &TreeStore<TRSMCTS<'ctx, 'b>>,
    children: bool,
) -> (Vec<Move<'ctx>>, f64) {
    let path = tree.path_tree(parent);
    let mut moves = path.iter().map(|mh| tree.mv(*mh).mov.clone()).collect_vec();
    moves.push(mv.clone());
    // Collect cost of choices so far.
    let mut score = path
        .iter()
        .map(|mh| -(tree.siblings_tree(*mh).len() as f64).ln())
        .sum();
    // Add cost of final choice.
    score -= (tree.node(parent).outgoing.len() as f64).ln();
    // Leave reward at each node.
    score -= ((moves.len() + (children as usize)) as f64) * 2f64.ln();
    (moves, score)
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

impl<'a, 'b> NodeStatistic<TRSMCTS<'a, 'b>> for VNMax {
    fn new() -> Self {
        VNMax {
            scores: Vec::with_capacity(11),
            n: 0.0,
        }
    }
    fn update(&mut self, evaluation: f64) {
        self.scores.push(evaluation);
        if self.scores.len() >= self.scores.capacity() {
            self.scores
                .sort_by(|a, b| a.partial_cmp(b).expect("No NAN"));
            self.scores.reverse();
            while self.scores.len() >= self.scores.capacity() {
                self.scores.pop();
            }
        }
        self.n += 1.0;
    }
    fn combine(&mut self, _other: &Self) {
        unimplemented!();
    }
}

impl<'a, 'b> NodeStatistic<TRSMCTS<'a, 'b>> for QNMean {
    fn new() -> Self {
        QNMean(QN {
            q: NEG_INFINITY,
            n: 0.0,
        })
    }
    fn update(&mut self, evaluation: f64) {
        self.0.q += evaluation;
        self.0.n += 1.0;
    }
    fn combine(&mut self, other: &Self) {
        self.0.q = logsumexp(&[self.0.q, other.0.q]);
    }
}

impl<'a, 'b> NodeStatistic<TRSMCTS<'a, 'b>> for QNN {
    fn new() -> Self {
        QNN(QN {
            q: NEG_INFINITY,
            n: 0.0,
        })
    }
    fn update(&mut self, _evaluation: f64) {
        self.0.n += 1.0;
    }
    fn combine(&mut self, _other: &Self) {}
}

impl<'a, 'b> NodeStatistic<TRSMCTS<'a, 'b>> for QNMax {
    fn new() -> Self {
        QNMax(QN {
            q: NEG_INFINITY,
            n: 0.0,
        })
    }
    fn update(&mut self, evaluation: f64) {
        self.0.q = self.0.q.max(evaluation);
        self.0.n += 1.0;
    }
    fn combine(&mut self, other: &Self) {
        self.0.q = self.0.q.max(other.0.q);
    }
}

impl<'a, 'b> NodeStatistic<TRSMCTS<'a, 'b>> for Vec<f64> {
    fn new() -> Self {
        Vec::new()
    }
    fn update(&mut self, evaluation: f64) {
        self.push(evaluation);
    }
    fn combine(&mut self, other: &Self) {
        self.extend_from_slice(other)
    }
}

//impl<'ctx> Move<'ctx> {
//    fn head(&self) -> String {
//        match *self {
//            Move::MemorizeAll => "MemorizeAll".to_string(),
//            Move::SampleAtom(_) => "SampleAtom".to_string(),
//            Move::RegenerateThisRule(..) => "RegenerateThisRule".to_string(),
//            Move::RegenerateThisPlace(..) => "RegenerateThisPlace".to_string(),
//            Move::Variablize(_) => "Variablize".to_string(),
//            Move::DeleteRule(_) => "DeleteRule".to_string(),
//            Move::MemorizeDatum(_) => "MemorizeDatum".to_string(),
//            Move::SampleRule => "SampleRule".to_string(),
//            Move::RegenerateRule => "RegenerateRule".to_string(),
//            Move::Generalize => "Generalize".to_string(),
//            Move::AntiUnify => "AntiUnify".to_string(),
//            Move::Compose(_) => "Compose".to_string(),
//            Move::Recurse(_) => "Recurse".to_string(),
//            Move::Stop => "Stop".to_string(),
//        }
//    }
//    fn pretty(&self, lex: &Lexicon) -> String {
//        match *self {
//            Move::MemorizeAll => "MemorizeAll".to_string(),
//            Move::MemorizeDatum(Some(n)) => format!("MemorizeDatum({})", n),
//            Move::SampleAtom(atom) => format!(
//                "SampleAtom({:?})",
//                atom.map(|atom| atom.display(lex.signature()))
//            ),
//            Move::RegenerateThisRule(n) => format!("RegenerateThisRule({})", n),
//            Move::RegenerateThisPlace(n) => format!("RegenerateThisPlace({:?})", n),
//            Move::DeleteRule(Some(n)) => format!("DeleteRule({})", n),
//            Move::Variablize(Some(ref v)) => format!("Variablize({}, {}, {:?})", v.0, v.1, v.2),
//            Move::Compose(Some(ref c)) => format!(
//                "Compose({}, {:?}, {:?}, {})",
//                c.0.pretty(lex.signature()),
//                c.1,
//                c.2,
//                c.3,
//            ),
//            Move::Recurse(Some(ref r)) => format!(
//                "Recurse({}, {:?}, {:?}, {})",
//                r.0.pretty(lex.signature()),
//                r.1,
//                r.2,
//                r.3,
//            ),
//            Move::Variablize(None) => "Variablize".to_string(),
//            Move::DeleteRule(None) => "DeleteRule".to_string(),
//            Move::MemorizeDatum(None) => "MemorizeDatum".to_string(),
//            Move::SampleRule => "SampleRule".to_string(),
//            Move::RegenerateRule => "RegenerateRule".to_string(),
//            Move::Generalize => "Generalize".to_string(),
//            Move::AntiUnify => "AntiUnify".to_string(),
//            Move::Compose(None) => "Compose".to_string(),
//            Move::Recurse(None) => "Recurse".to_string(),
//            Move::Stop => "Stop".to_string(),
//        }
//    }
//}

impl<'ctx> std::fmt::Display for Move<'ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Move::MemorizeAll => write!(f, "MemorizeAll"),
            Move::MemorizeDatum(Some(n)) => write!(f, "MemorizeDatum({})", n),
            Move::Variablize(Some(ref v)) => write!(f, "Variablize({}, {}, {:?})", v.0, v.1, v.2),
            Move::SampleAtom(atom) => write!(f, "SampleAtom({:?})", atom),
            Move::RegenerateThisRule(n) => write!(f, "RegenerateThisRule({})", n),
            Move::RegenerateThisPlace(n) => write!(f, "RegenerateThisPlace({:?})", n),
            Move::DeleteRule(Some(n)) => write!(f, "DeleteRule({})", n),
            Move::MemorizeDatum(None) => write!(f, "MemorizeDatum"),
            Move::SampleRule => write!(f, "SampleRule"),
            Move::RegenerateRule => write!(f, "RegenerateRule"),
            Move::DeleteRule(None) => write!(f, "DeleteRule"),
            Move::Generalize => write!(f, "Generalize"),
            Move::AntiUnify => write!(f, "AntiUnify"),
            Move::Variablize(None) => write!(f, "Variablize"),
            Move::Compose(None) => write!(f, "Compose"),
            Move::Recurse(None) => write!(f, "Recurse"),
            Move::Compose(_) => write!(f, "Compose(_)"),
            Move::Recurse(_) => write!(f, "Recurse(_)"),
            Move::Stop => write!(f, "Stop"),
        }
    }
}

impl Terminal {
    pub fn new(trs: HypothesisHandle) -> Self {
        Terminal { trs }
    }
}

impl Revision {
    pub fn new() -> Self {
        let playout = PlayoutState::Untried;
        Revision { playout }
    }
    pub fn show(&self) {
        println!("playout: {:?}", self.playout);
    }
}

impl Default for Revision {
    fn default() -> Self {
        Self::new()
    }
}

impl<'ctx, 'b> State<TRSMCTS<'ctx, 'b>> for MCTSState {
    type Data = TrueState<'ctx, 'b>;
    type Move = Move<'ctx>;
    type MoveList = Vec<Self::Move>;
    fn root_data(mcts: &TRSMCTS<'ctx, 'b>) -> Self::Data {
        TrueState {
            trs: TRS::new_unchecked(&mcts.lexicon, mcts.deterministic, mcts.bg, vec![]),
            spec: None,
            n: 0,
            path: vec![],
            label: StateLabel::CompleteRevision,
        }
    }
    fn valid_data(data: &Self::Data, _mcts: &TRSMCTS<'ctx, 'b>) -> bool {
        data.label != StateLabel::Failed
    }
    fn available_moves(
        &self,
        data: &mut Self::Data,
        depth: usize,
        mcts: &TRSMCTS<'ctx, 'b>,
    ) -> Self::MoveList {
        match *self {
            MCTSState::Terminal(_) => vec![],
            MCTSState::Revision(_) if mcts.max_depth() <= depth => vec![],
            MCTSState::Revision(_) => data.available_moves(mcts),
        }
    }
    fn make_move(
        &self,
        data: &mut Self::Data,
        mv: &Self::Move,
        n: usize,
        mcts: &TRSMCTS<'ctx, 'b>,
    ) {
        match *self {
            MCTSState::Terminal(_) => panic!("cannot move from terminal"),
            MCTSState::Revision(_) => data.make_move(mv, n, mcts.data),
        }
    }
    fn make_state(data: &Self::Data, mcts: &mut TRSMCTS<'ctx, 'b>) -> Option<Self> {
        match data.label {
            StateLabel::Failed => None,
            StateLabel::Terminal => {
                let terminal = Terminal::new(mcts.make_hypothesis(data));
                Some(mcts.add_terminal(terminal))
            }
            StateLabel::PartialRevision => Some(mcts.add_revision(Revision::default())),
            StateLabel::CompleteRevision => Some(mcts.add_revision(Revision::default())),
        }
    }
    //fn describe_self(&self, data: &Self::Data, mcts: &TRSMCTS) -> Value {
    //    match *self {
    //        MCTSState::Terminal(th) => {
    //            let hh = mcts.terminals[th].trs;
    //            let trs = &mcts.hypotheses[hh].object.trs;
    //            let trs_string = trs.utrs.pretty(trs.lex.signature());
    //            json!({
    //                "type": "terminal",
    //                "trs": trs_string,
    //            })
    //        }
    //        MCTSState::Revision(rh) => {
    //            let trs_string = data.trs.utrs.pretty(data.trs.lex.signature());
    //            let playout_string = match mcts.revisions[rh].playout {
    //                PlayoutState::Failed => "failed".to_string(),
    //                PlayoutState::Untried => "untried".to_string(),
    //                PlayoutState::Success(hh) => {
    //                    let playout = &mcts.hypotheses[hh].object.trs;
    //                    playout.utrs.pretty(playout.lex.signature())
    //                }
    //            };
    //            json!({
    //                "type": "revision",
    //                "n": data.n,
    //                "trs": trs_string,
    //                "playout": playout_string,
    //            })
    //        }
    //    }
    //}
    //fn describe_move(
    //    &self,
    //    data: &Self::Data,
    //    mv: &Self::Move,
    //    _mcts: &TRSMCTS,
    //    failed: bool,
    //) -> Value {
    //    match *self {
    //        MCTSState::Terminal(_) => Value::Null,
    //        MCTSState::Revision(_) => {
    //            if failed {
    //                Value::String(mv.head())
    //            } else {
    //                Value::String(mv.pretty(&data.trs.lex))
    //            }
    //        }
    //    }
    //}
    fn discard(&self, mcts: &mut TRSMCTS<'ctx, 'b>) {
        mcts.rm_state(self)
    }
}

impl<'ctx, 'b> TrueState<'ctx, 'b> {
    pub fn playout<R: Rng>(&self, mcts: &mut TRSMCTS<'ctx, 'b>, rng: &mut R) -> Option<Self> {
        // TODO: Maybe try backtracking instead of a fixed count?
        for _ in 0..10 {
            let mut state = self.clone();
            let mut progress = true;
            let mut depth = state.path.len();
            while progress && depth < mcts.params.max_depth {
                progress = false;
                // Compute the available moves.
                let mut moves = state.available_moves(mcts);
                // Choose a move (random policy favoring STOP).
                let moves_len = moves.len();
                moves.shuffle(rng);
                if let Some(idx) = moves.iter().position(|mv| *mv == Move::Stop) {
                    moves.swap(0, idx);
                }
                let old_state = state.clone();
                for mv in moves {
                    state.make_move(&mv, moves_len, mcts.data);
                    match state.label {
                        StateLabel::Failed => {
                            state = old_state.clone();
                            continue;
                        }
                        StateLabel::Terminal => return Some(state),
                        _ => {
                            depth += 1;
                            progress = true;
                            break;
                        }
                    }
                }
            }
        }
        None
    }
    fn available_moves(&mut self, mcts: &TRSMCTS<'ctx, 'b>) -> Vec<Move<'ctx>> {
        let mut moves = vec![];
        match &mut self.spec {
            None => {
                // Search can always stop.
                moves.push(Move::Stop);
                if self.n < mcts.params.max_revisions {
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
                    if mcts.data.iter().any(|datum| match datum {
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
                (0..mcts.data.len())
                    .filter(|idx| match mcts.data[*idx] {
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
                self.path.push((mv.clone(), n));
                self.spec = None;
                self.label = StateLabel::Terminal;
            }
            Move::Generalize => {
                self.trs = tryo![self, self.trs.generalize().ok()];
                self.n += 1;
                self.path.push((mv.clone(), n));
                self.spec = None;
                self.label = StateLabel::CompleteRevision;
            }
            Move::AntiUnify => {
                self.trs = tryo![self, self.trs.lgg().ok()];
                self.n += 1;
                self.path.push((mv.clone(), n));
                self.spec = None;
                self.label = StateLabel::CompleteRevision;
            }
            Move::Compose(None) => {
                self.path.push((mv.clone(), n));
                self.label = StateLabel::PartialRevision;
                self.spec.replace(MoveState::Compose);
            }
            Move::Compose(Some(ref composition)) => {
                self.trs = tryo![self, self.trs.compose_by(composition)];
                self.n += 1;
                self.path.push((mv.clone(), n));
                self.spec = None;
                self.label = StateLabel::CompleteRevision;
            }
            Move::Recurse(None) => {
                self.path.push((mv.clone(), n));
                self.label = StateLabel::PartialRevision;
                self.spec.replace(MoveState::Recurse);
            }
            Move::Recurse(Some(ref recursion)) => {
                self.trs = tryo![self, self.trs.recurse_by(recursion)];
                self.n += 1;
                self.path.push((mv.clone(), n));
                self.spec = None;
                self.label = StateLabel::CompleteRevision;
            }
            Move::Variablize(None) => {
                self.path.push((mv.clone(), n));
                self.label = StateLabel::PartialRevision;
                self.spec.replace(MoveState::Variablize);
            }
            Move::Variablize(Some(ref v)) => {
                let mut clauses = self.trs.utrs.clauses();
                if clauses.len() <= v.0 {
                    self.label = StateLabel::Failed;
                    return;
                }
                clauses[v.0] = tryo![
                    self,
                    self.trs.apply_variablization(&v.1, &v.2, &clauses[v.0])
                ];
                // TODO: remove clone
                self.trs = tryo![self, self.trs.clone().adopt_rules(&mut clauses)];
                self.n += 1;
                self.path.push((mv.clone(), n));
                self.spec = None;
                self.label = StateLabel::CompleteRevision;
            }
            Move::DeleteRule(None) => {
                self.path.push((mv.clone(), n));
                self.label = StateLabel::PartialRevision;
                self.spec.replace(MoveState::DeleteRule);
            }
            Move::DeleteRule(Some(r)) => {
                tryo![self, self.trs.utrs.remove_idx(r).ok()];
                self.n += 1;
                self.path.push((mv.clone(), n));
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
                    self.path.push((mv.clone(), n));
                    self.spec = None;
                    self.label = StateLabel::CompleteRevision;
                }
            }
            Move::MemorizeDatum(None) => {
                self.path.push((mv.clone(), n));
                self.label = StateLabel::PartialRevision;
                self.spec.replace(MoveState::MemorizeDatum);
            }
            Move::MemorizeDatum(Some(r)) => match data[r] {
                Datum::Partial(_) => panic!("can't memorize partial data"),
                Datum::Full(ref rule) => {
                    tryo![self, self.trs.append_clauses(vec![rule.clone()]).ok()];
                    self.n += 1;
                    self.path.push((mv.clone(), n));
                    self.spec = None;
                    self.label = StateLabel::CompleteRevision;
                }
            },
            Move::SampleRule => {
                self.path.push((mv.clone(), n));
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
                self.path.push((mv.clone(), n));
                self.label = StateLabel::PartialRevision;
                self.spec
                    .replace(MoveState::RegenerateRule(RegenerateRuleState::Start));
            }
            Move::RegenerateThisRule(r) => {
                if self.trs.utrs.rules.len() <= r {
                    self.label = StateLabel::Failed;
                } else {
                    self.path.push((mv.clone(), n));
                    self.label = StateLabel::PartialRevision;
                    self.spec
                        .replace(MoveState::RegenerateRule(RegenerateRuleState::Rule(r)));
                }
            }
            Move::RegenerateThisPlace(p) => {
                self.path.push((mv.clone(), n));
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
                                self.path.push((mv.clone(), n));
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
                                self.path.push((mv.clone(), n));
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
                                self.path.push((mv.clone(), n));
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
                                self.path.push((mv.clone(), n));
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

impl<'ctx, 'b> MCTS for TRSMCTS<'ctx, 'b> {
    type StateEval = MCTSStateEvaluator;
    type MoveEval = MaxThompsonMoveEvaluator;
    type State = MCTSState;
    fn max_depth(&self) -> usize {
        self.params.max_depth
    }
    fn max_states(&self) -> usize {
        self.params.max_states
    }
}

impl<'ctx, 'b> TRSMCTS<'ctx, 'b> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lexicon: Lexicon<'ctx, 'b>,
        bg: &'b [Rule],
        deterministic: bool,
        lo: usize,
        hi: usize,
        data: &'b [&'b Datum],
        model: ModelParams,
        params: MCTSParams,
    ) -> TRSMCTS<'ctx, 'b> {
        TRSMCTS {
            lexicon,
            bg,
            deterministic,
            lo,
            hi,
            data,
            model,
            params,
            root: None,
            hypotheses: Arena::new(),
            terminals: Arena::new(),
            revisions: Arena::new(),
            best: NEG_INFINITY,
            search_time: 0.0,
            trial_start: None,
            count: 0,
        }
    }
    pub fn start_trial(&mut self) {
        self.trial_start.replace(std::time::Instant::now());
    }
    pub fn finish_trial(&mut self) {
        // Stop trial.
        self.search_time += self
            .trial_start
            .map(|ts| ts.elapsed().as_secs_f64())
            .unwrap_or(0.0);
    }
    pub fn rm_state(&mut self, state: &MCTSState) {
        match *state {
            MCTSState::Terminal(h) => self.rm_terminal(h),
            MCTSState::Revision(r) => self.rm_revision(r),
        }
    }
    pub fn add_revision(&mut self, state: Revision) -> MCTSState {
        MCTSState::Revision(RevisionHandle(self.revisions.insert(state)))
    }
    pub fn rm_revision(&mut self, rh: RevisionHandle) {
        if let PlayoutState::Success(hh) = self.revisions[rh].playout {
            self.rm_hypothesis(hh);
        }
        self.revisions.remove(rh.0);
    }
    pub fn add_terminal(&mut self, state: Terminal) -> MCTSState {
        MCTSState::Terminal(TerminalHandle(self.terminals.insert(state)))
    }
    pub fn rm_terminal(&mut self, th: TerminalHandle) {
        self.rm_hypothesis(self.terminals[th].trs);
        self.terminals.remove(th.0);
    }
    pub fn find_trs(trs: &mut TRS<'ctx, 'b>) {
        trs.utrs.canonicalize(&mut HashMap::new());
    }
    pub fn make_hypothesis(&mut self, state: &TrueState<'ctx, 'b>) -> HypothesisHandle {
        let time = self.search_time
            + self
                .trial_start
                .map(|ts| ts.elapsed().as_secs_f64())
                .unwrap_or(0.0);

        let count = self.count;
        self.count += 1;

        let moves = state.path.iter().fold(
            Vec::with_capacity(state.path.len()),
            |mut moves, (mv, _)| {
                moves.push(mv.clone());
                moves
            },
        );

        let mut truestate = state.clone();
        truestate.trs.utrs.canonicalize(&mut HashMap::new());
        let meta_program_prior = truestate.path.iter().fold(0.0, |partial_prior, (_, n)| {
            partial_prior - (*n as f64).ln()
        });
        let trs_prior = truestate.trs.log_prior(self.model.prior);
        let mut l1 = self.model.likelihood;
        l1.single = SingleLikelihood::Generalization(0.001);
        let soft_generalization_likelihood = truestate.trs.log_likelihood(self.data, l1);
        let mut l2 = self.model.likelihood;
        l2.single = SingleLikelihood::Generalization(0.0);
        let hard_generalization_likelihood = truestate.trs.log_likelihood(self.data, l2);
        let accuracy_likelihood = truestate
            .trs
            .log_likelihood(self.data, self.model.likelihood);
        // Noisy-OR
        let ln_search_prior = logdiffexp(
            logsumexp(&[meta_program_prior, trs_prior]),
            meta_program_prior + trs_prior,
        );
        let ln_search_likelihood = accuracy_likelihood + soft_generalization_likelihood;
        let ln_search_posterior =
            ln_search_prior * self.model.p_temp + ln_search_likelihood * self.model.l_temp;
        // After HL finds a meta-program, it doesn't care how it found it.
        let ln_predict_prior = trs_prior;
        let ln_predict_likelihood = accuracy_likelihood + hard_generalization_likelihood;
        let ln_predict_posterior =
            ln_predict_prior * self.model.p_temp + ln_predict_likelihood * self.model.l_temp;
        let object = MCTSObj::new(
            time,
            count,
            moves,
            meta_program_prior,
            trs_prior,
            accuracy_likelihood,
            hard_generalization_likelihood,
            ln_search_prior,
            ln_search_likelihood,
            ln_search_posterior,
            ln_predict_prior,
            ln_predict_likelihood,
            ln_predict_posterior,
        );
        HypothesisHandle(self.hypotheses.insert(Box::new(object)))
    }
    pub fn rm_hypothesis(&mut self, hh: HypothesisHandle) {
        self.hypotheses.remove(hh.0);
    }
    pub fn clear(&mut self) {
        self.hypotheses.clear();
        self.terminals.clear();
        self.revisions.clear();
        self.best = NEG_INFINITY;
        self.root = None;
    }
    pub fn root(&mut self) -> MCTSState {
        let mut trs = TRS::new_unchecked(&self.lexicon, self.deterministic, self.bg, vec![]);
        trs.utrs.lo = self.lo;
        trs.utrs.hi = self.hi;
        trs.identify_symbols();
        let state = Revision {
            playout: PlayoutState::Untried,
        };
        let root_state = self.add_revision(state);
        self.root.replace(root_state);
        root_state
    }
    //pub fn update_hypotheses(&mut self) {
    //    let mut hypotheses = std::mem::replace(&mut self.hypotheses, Arena::new());
    //    for (_, hypothesis) in hypotheses.iter_mut() {
    //        let (new_trs, new_prior) = self.p_path(&hypothesis.object.meta);
    //        hypothesis.object.meta_prior = new_prior;
    //        if let Some(trs) = new_trs {
    //            hypothesis.object.trs = trs;
    //        }
    //        hypothesis.log_posterior(self.data);
    //    }
    //    self.hypotheses = hypotheses;
    //}
    //pub fn p_path(&self, moves: &[Move<'ctx>]) -> (Option<TRS<'ctx, 'b>>, f64) {
    //    // Set the root state.
    //    let mut trs = match self.root {
    //        Some(MCTSState(StateHandle::Revision(rh))) => self.revisions[rh].trs.clone(),
    //        _ => panic!("no root"),
    //    };
    //    let mut spec = None;
    //    let mut n = 0;
    //    let mut ps = Vec::with_capacity(moves.len());
    //    for mv in moves {
    //        // Ensure our move is available.
    //        let available_moves = Revision::available_moves_inner(&trs, &spec, n, self);
    //        if !available_moves.contains(mv) {
    //            return (None, NEG_INFINITY);
    //        }
    //        // Take the move and update the state accordingly.
    //        match Revision::make_move_inner(&mv, &spec, &self.data, &trs) {
    //            MoveResult::Failed => {
    //                return (None, NEG_INFINITY);
    //            }
    //            MoveResult::Terminal => {
    //                ps.push(available_moves.len() as f64);
    //                break;
    //            }
    //            MoveResult::Revision(new_trs, new_spec) => {
    //                ps.push(available_moves.len() as f64);
    //                n += new_spec.is_none() as usize;
    //                spec = new_spec;
    //                if let Some(new_trs) = new_trs {
    //                    trs = new_trs;
    //                }
    //            }
    //        }
    //    }
    //    if ps.len() == moves.len() {
    //        let prior = ps.into_iter().map(|p| -(2.0 * p).ln()).sum();
    //        (Some(trs), prior)
    //    } else {
    //        (None, NEG_INFINITY)
    //    }
    //}
    //fn update_playout(&mut self, rh: RevisionHandle) -> bool {
    //    match self.revisions[rh].playout {
    //        PlayoutState::Untried(..) => false,
    //        PlayoutState::Failed => true,
    //        PlayoutState::Success(hh) => {
    //            let (new_trs, new_prior) = self.p_path(&self.hypotheses[hh].object.meta);
    //            if let Some(trs) = new_trs {
    //                self.hypotheses[hh].object.meta_prior = new_prior;
    //                self.hypotheses[hh].object.trs = trs;
    //                false
    //            } else {
    //                true
    //            }
    //        }
    //    }
    //}
}

fn unexplored_first<'ctx, 'b, F, R: Rng, MoveIter>(
    moves: MoveIter,
    nh: NodeHandle,
    mcts: &TRSMCTS<'ctx, 'b>,
    tree: &TreeStore<TRSMCTS<'ctx, 'b>>,
    selector: F,
    rng: &mut R,
) -> Option<MoveHandle>
where
    MoveIter: Iterator<Item = MoveHandle>,
    F: Fn(NodeHandle, NodeHandle, &TreeStore<TRSMCTS<'ctx, 'b>>, &TRSMCTS<'ctx, 'b>, &mut R) -> f64,
{
    // Split the moves into those with and without children.
    let (childful, childless): (Vec<_>, Vec<_>) =
        moves.partition(|mh| tree.mv(*mh).child.is_some());
    // Take the first childless move, or perform UCT on childed moves.
    if let Some(mh) = childless.iter().find(|mh| tree.mv(**mh).mov == Move::Stop) {
        Some(*mh)
    } else if let Some(mh) = childless.choose(rng) {
        Some(*mh)
    } else {
        childful
            .into_iter()
            .map(|mh| {
                let ch = tree.mv(mh).child.expect("INVARIANT: partition failed us");
                let score = selector(nh, ch, tree, mcts, rng);
                (mh, score)
            })
            .fold(vec![], |mut acc, x| {
                if acc.is_empty() {
                    acc.push(x);
                    acc
                } else {
                    match acc[0].1.partial_cmp(&x.1) {
                        None | Some(Ordering::Greater) => acc,
                        Some(Ordering::Less) => vec![x],
                        Some(Ordering::Equal) => {
                            acc.push(x);
                            acc
                        }
                    }
                }
            })
            .choose(rng)
            .map(|(mv, _)| *mv)
            .or_else(|| None)
    }
}

//impl<'a, 'b> MoveEvaluator<TRSMCTS<'a, 'b>> for RescaledByBestMoveEvaluator {
//    type NodeStatistics = QNMean;
//    fn choose<'c, R: Rng, MoveIter>(
//        &self,
//        moves: MoveIter,
//        nh: NodeHandle,
//        tree: &SearchTree<TRSMCTS<'a, 'b>>,
//        rng: &mut R,
//    ) -> Option<&'c MoveInfo<TRSMCTS<'a, 'b>>>
//    where
//        MoveIter: Iterator<Item = &'c MoveInfo<TRSMCTS<'a, 'b>>>,
//    {
//        unexplored_first(moves, nh, tree, rescaled_by_best_uct, rng)
//    }
//}

//impl<'a, 'b> MoveEvaluator<TRSMCTS<'a, 'b>> for BestSoFarMoveEvaluator {
//    type NodeStatistics = QNMax;
//    fn choose<'c, R: Rng, MoveIter>(
//        &self,
//        moves: MoveIter,
//        nh: NodeHandle,
//        tree: &SearchTree<TRSMCTS<'a, 'b>>,
//        rng: &mut R,
//    ) -> Option<&'c MoveInfo<TRSMCTS<'a, 'b>>>
//    where
//        MoveIter: Iterator<Item = &'c MoveInfo<TRSMCTS<'a, 'b>>>,
//    {
//        unexplored_first(moves, nh, tree, best_so_far_uct, rng)
//    }
//}

//impl<'a, 'b> MoveEvaluator<TRSMCTS<'a, 'b>> for ThompsonMoveEvaluator {
//    type NodeStatistics = QNN;
//    fn choose<R: Rng, MoveIter>(
//        &self,
//        moves: MoveIter,
//        nh: NodeHandle,
//        tree: &SearchTree<TRSMCTS<'a, 'b>>,
//        rng: &mut R,
//    ) -> Option<MoveHandle>
//    where
//        MoveIter: Iterator<Item = MoveHandle>,
//    {
//        unexplored_first(moves, nh, tree.mcts(), tree.tree(), thompson_sample, rng)
//    }
//}

impl<'a, 'b> MoveEvaluator<TRSMCTS<'a, 'b>> for MaxThompsonMoveEvaluator {
    type NodeStatistics = VNMax;
    fn choose<R: Rng, MoveIter>(
        &self,
        moves: MoveIter,
        nh: NodeHandle,
        tree: &SearchTree<TRSMCTS<'a, 'b>>,
        rng: &mut R,
    ) -> Option<MoveHandle>
    where
        MoveIter: Iterator<Item = MoveHandle>,
    {
        unexplored_first(
            moves,
            nh,
            tree.mcts(),
            tree.tree(),
            max_thompson_sample,
            rng,
        )
    }
}

impl<'a, 'b> StateEvaluator<TRSMCTS<'a, 'b>> for MCTSStateEvaluator {
    type StateEvaluation = f64;
    fn reread(
        &self,
        state: &<TRSMCTS<'a, 'b> as MCTS>::State,
        mcts: &mut TRSMCTS<'a, 'b>,
    ) -> Self::StateEvaluation {
        match *state {
            MCTSState::Terminal(th) => mcts.hypotheses[mcts.terminals[th].trs].ln_search_posterior,
            MCTSState::Revision(rh) => match mcts.revisions[rh].playout {
                PlayoutState::Untried => panic!("shouldn't reread untried playout"),
                PlayoutState::Failed => NEG_INFINITY,
                PlayoutState::Success(hh) => mcts.hypotheses[hh].ln_search_posterior,
            },
        }
    }
    fn evaluate<R: Rng>(
        &self,
        state: &<TRSMCTS<'a, 'b> as MCTS>::State,
        data: &<<TRSMCTS<'a, 'b> as MCTS>::State as State<TRSMCTS<'a, 'b>>>::Data,
        mcts: &mut TRSMCTS<'a, 'b>,
        rng: &mut R,
    ) -> Self::StateEvaluation {
        let score = match *state {
            MCTSState::Terminal(th) => mcts.hypotheses[mcts.terminals[th].trs].ln_search_posterior,
            MCTSState::Revision(rh) => match mcts.revisions[rh].playout {
                // Note: the DAG creates multiple nodes sharing the same state.
                // They share a single playout, which may not be correct.
                PlayoutState::Failed => NEG_INFINITY,
                PlayoutState::Success(hh) => mcts.hypotheses[hh].ln_search_posterior,
                PlayoutState::Untried => match data.playout(mcts, rng) {
                    Some(state) => {
                        let hh = mcts.make_hypothesis(&state);
                        mcts.revisions[rh].playout = PlayoutState::Success(hh);
                        mcts.hypotheses[hh].ln_search_posterior
                    }
                    None => {
                        mcts.revisions[rh].playout = PlayoutState::Failed;
                        NEG_INFINITY
                    }
                },
            },
        };
        if score > mcts.best {
            mcts.best = score;
        }
        score
    }
}

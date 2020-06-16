use generational_arena::{Arena, Index};
use itertools::Itertools;
use mcts::{
    MoveEvaluator, MoveHandle, NodeHandle, NodeStatistic, SearchTree, State, StateEvaluator,
    TreeStore, MCTS,
};
use polytype::atype::Ty;
use rand::{
    distributions::Bernoulli,
    prelude::{Distribution, Rng, SliceRandom},
};
use serde_json::Value;
use std::{cmp::Ordering, collections::HashMap, convert::TryFrom};
use term_rewriting::{Atom, Context, Rule, RuleContext, SituatedAtom, Term};
use trs::{
    Composition, Datum, Env, Eval, Hypothesis, Lexicon, ModelParams, ProbabilisticModel, Recursion,
    Variablization, TRS,
};
use utils::logsumexp;

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

type Hyp<'ctx, 'b> = Hypothesis<MCTSObj<'ctx, 'b>, &'b Datum, MCTSModel<'ctx, 'b>>;

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

#[derive(Debug, Clone)]
pub enum PlayoutState<T: std::fmt::Debug + Copy> {
    Untried,
    Failed,
    Success(T),
}

pub struct BestSoFarMoveEvaluator;
pub struct RescaledByBestMoveEvaluator;
pub struct RelativeMassMoveEvaluator;
pub struct ThompsonMoveEvaluator;
pub struct MCTSStateEvaluator;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum MoveState<'ctx, 'b> {
    SampleRule(RuleContext, Env<'ctx, 'b>, Vec<Ty<'ctx>>),
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
    Term(usize, RuleContext, Env<'ctx, 'b>, Vec<Ty<'ctx>>),
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
    Variablize(Option<Variablization<'ctx>>),
    Compose(Option<Composition<'ctx>>),
    Recurse(Option<Recursion<'ctx>>),
    MemorizeAll,
    Generalize,
    AntiUnify,
    Stop,
}

#[derive(Debug, PartialEq, Eq, Clone)]
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
    pub hypotheses: Arena<Hypothesis<MCTSObj<'ctx, 'b>, &'b Datum, MCTSModel<'ctx, 'b>>>,
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

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum Selection {
    RelativeMassUCT,
    BestSoFarUCT,
    Thompson(u32),
}

pub struct MCTSObj<'ctx, 'b> {
    pub trs: TRS<'ctx, 'b>,
    pub time: f64,
    pub count: usize,
    pub meta: Vec<Move<'ctx>>,
    pub meta_prior: f64,
}

pub struct MCTSModel<'a, 'b> {
    params: ModelParams,
    evals: HashMap<&'b Datum, Eval>,
    phantom: std::marker::PhantomData<TRS<'a, 'b>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TrueState<'ctx, 'b> {
    trs: TRS<'ctx, 'b>,
    n: usize,
    spec: Option<MoveState<'ctx, 'b>>,
    path: Vec<(Move<'ctx>, usize)>,
    label: StateLabel,
}

impl<'a, 'b> MCTSModel<'a, 'b> {
    pub fn new(params: ModelParams) -> Self {
        MCTSModel {
            params,
            evals: HashMap::new(),
            phantom: std::marker::PhantomData,
        }
    }
    pub fn generalizes(&self, data: &[Datum]) -> bool {
        data.iter().all(|datum| {
            self.evals
                .get(datum)
                .map(|eval| eval.generalizes())
                .unwrap_or(false)
        })
    }
}

impl<'ctx, 'b> MCTSObj<'ctx, 'b> {
    pub fn new(
        trs: TRS<'ctx, 'b>,
        time: f64,
        count: usize,
        meta: Vec<Move<'ctx>>,
        meta_prior: f64,
    ) -> Self {
        MCTSObj {
            trs,
            time,
            count,
            meta,
            meta_prior,
        }
    }
}

impl<'ctx, 'b> std::ops::Index<HypothesisHandle> for Arena<Hyp<'ctx, 'b>> {
    type Output = Hyp<'ctx, 'b>;
    fn index(&self, index: HypothesisHandle) -> &Self::Output {
        &self[index.0]
    }
}

impl<'ctx, 'b> std::ops::IndexMut<HypothesisHandle> for Arena<Hyp<'ctx, 'b>> {
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

impl<'a, 'b> ProbabilisticModel for MCTSModel<'a, 'b> {
    type Object = MCTSObj<'a, 'b>;
    type Datum = &'b Datum;
    fn log_prior(&mut self, object: &Self::Object) -> f64 {
        object.meta_prior
    }
    fn single_log_likelihood<DataIter>(
        &mut self,
        object: &Self::Object,
        data: &Self::Datum,
    ) -> f64 {
        object
            .trs
            .single_log_likelihood(data, self.params.likelihood)
            .likelihood()
    }
    fn log_likelihood(&mut self, object: &Self::Object, data: &[Self::Datum]) -> f64 {
        object
            .trs
            .log_likelihood(data, &mut self.evals, self.params.likelihood)
    }
    fn log_posterior(&mut self, object: &Self::Object, data: &[Self::Datum]) -> (f64, f64, f64) {
        let lprior = self.log_prior(object);
        let llikelihood = self.log_likelihood(object, data);
        let lposterior = lprior * self.params.p_temp + llikelihood * self.params.l_temp;
        (lprior, llikelihood, lposterior)
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

pub fn thompson_sample<R: Rng>(
    parent: NodeHandle,
    child: NodeHandle,
    tree: &TreeStore<TRSMCTS>,
    mcts: &TRSMCTS,
    rng: &mut R,
) -> f64 {
    let mut source = {
        let a = match mcts.params.selection {
            Selection::Thompson(a) => a,
            x => panic!("in thompson_sample but Selection is {:?}", x),
        };
        let b = tree.node(child).stats.0.n as u32;
        if Bernoulli::from_ratio(b, a + b).unwrap().sample(rng) {
            child
        } else {
            parent
        }
    };
    let mut source_node = tree.node(source);
    let mut n = rng.gen_range(0, source_node.stats.0.n as usize);
    let mut outgoing = &source_node.outgoing;
    let mut idx = 0;
    while n != 0 {
        match tree.mv(outgoing[idx]).child {
            None => idx += 1,
            Some(target) => {
                let target_node = tree.node(target);
                if n > target_node.stats.0.n as usize {
                    idx += 1;
                    n -= target_node.stats.0.n as usize;
                } else {
                    source = target;
                    source_node = target_node;
                    outgoing = &source_node.outgoing;
                    idx = 0;
                    n -= 1;
                }
            }
        }
    }
    tree.node(source).evaluation
}

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
///   let fillers = rulecontext_fillers(&lex, &context);
///   println!("{}", fillers.len());
///   for a in fillers {
///       println!("{}", a.display(lex.signature()));
///   }
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
            let tp = arg_types[0].apply(&env.sub);
            env.enumerate_atoms(tp)
                .filter_map(|atom| match atom {
                    None if full_lhs_hole => None,
                    Some(a) if full_lhs_hole && a.is_variable() => None,
                    x => Some(x),
                })
                .collect_vec()
        }
    }
}

impl<'a, 'b> NodeStatistic<TRSMCTS<'a, 'b>> for QNMean {
    fn new() -> Self {
        QNMean(QN {
            q: std::f64::NEG_INFINITY,
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
            q: std::f64::NEG_INFINITY,
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
            q: std::f64::NEG_INFINITY,
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

impl<'ctx> Move<'ctx> {
    fn head(&self) -> String {
        match *self {
            Move::MemorizeAll => "MemorizeAll".to_string(),
            Move::SampleAtom(_) => "SampleAtom".to_string(),
            Move::RegenerateThisRule(..) => "RegenerateThisRule".to_string(),
            Move::RegenerateThisPlace(..) => "RegenerateThisPlace".to_string(),
            Move::Variablize(_) => "Variablize".to_string(),
            Move::DeleteRule(_) => "DeleteRule".to_string(),
            Move::MemorizeDatum(_) => "MemorizeDatum".to_string(),
            Move::SampleRule => "SampleRule".to_string(),
            Move::RegenerateRule => "RegenerateRule".to_string(),
            Move::Generalize => "Generalize".to_string(),
            Move::AntiUnify => "AntiUnify".to_string(),
            Move::Compose(_) => "Compose".to_string(),
            Move::Recurse(_) => "Recurse".to_string(),
            Move::Stop => "Stop".to_string(),
        }
    }
    fn pretty(&self, lex: &Lexicon) -> String {
        match *self {
            Move::MemorizeAll => "MemorizeAll".to_string(),
            Move::MemorizeDatum(Some(n)) => format!("MemorizeDatum({})", n),
            Move::SampleAtom(atom) => format!(
                "SampleAtom({:?})",
                atom.map(|atom| atom.display(lex.signature()))
            ),
            Move::RegenerateThisRule(n) => format!("RegenerateThisRule({})", n),
            Move::RegenerateThisPlace(n) => format!("RegenerateThisPlace({:?})", n),
            Move::DeleteRule(Some(n)) => format!("DeleteRule({})", n),
            Move::Variablize(Some((ref n, ref t, ref ps))) => {
                format!("Variablize({}, {}, {:?})", n, t, ps)
            }
            Move::Compose(Some((ref t, ref p1, ref p2, ref tp))) => format!(
                "Compose({}, {:?}, {:?}, {})",
                t.pretty(lex.signature()),
                p1,
                p2,
                tp
            ),
            Move::Recurse(Some((ref t, ref p1, ref p2, ref tp))) => format!(
                "Recurse({}, {:?}, {:?}, {})",
                t.pretty(lex.signature()),
                p1,
                p2,
                tp
            ),
            Move::Variablize(None) => "Variablize".to_string(),
            Move::DeleteRule(None) => "DeleteRule".to_string(),
            Move::MemorizeDatum(None) => "MemorizeDatum".to_string(),
            Move::SampleRule => "SampleRule".to_string(),
            Move::RegenerateRule => "RegenerateRule".to_string(),
            Move::Generalize => "Generalize".to_string(),
            Move::AntiUnify => "AntiUnify".to_string(),
            Move::Compose(None) => "Compose".to_string(),
            Move::Recurse(None) => "Recurse".to_string(),
            Move::Stop => "Stop".to_string(),
        }
    }
}

impl<'ctx> std::fmt::Display for Move<'ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Move::MemorizeAll => write!(f, "MemorizeAll"),
            Move::MemorizeDatum(Some(n)) => write!(f, "MemorizeDatum({})", n),
            Move::Variablize(Some((ref n, ref t, ref ps))) => {
                write!(f, "Variablize({}, {}, {:?})", n, t, ps)
            }
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
    fn available_moves(&self, data: &mut Self::Data, mcts: &TRSMCTS<'ctx, 'b>) -> Self::MoveList {
        match *self {
            MCTSState::Terminal(_) => vec![],
            MCTSState::Revision(_) => data.available_moves(mcts),
        }
    }
    fn make_move(
        &self,
        data: &mut Self::Data,
        mv: &Self::Move,
        n: usize,
        mcts: &mut TRSMCTS<'ctx, 'b>,
    ) {
        match *self {
            MCTSState::Terminal(_) => panic!("cannot move from terminal"),
            MCTSState::Revision(_) => data.make_move(mv, n, mcts.data),
        }
    }
    fn make_state(data: &Self::Data, mcts: &mut TRSMCTS<'ctx, 'b>) -> Option<Self> {
        // TODO: implement
        match data.label {
            StateLabel::Failed => None,
            StateLabel::Terminal => {
                let terminal = Terminal::new(mcts.make_hypothesis(data));
                Some(mcts.add_terminal(terminal))
            }
            StateLabel::PartialRevision => Some(mcts.add_revision(Revision::new())),
            StateLabel::CompleteRevision => Some(mcts.add_revision(Revision::new())),
        }
    }
    fn describe_self(&self, data: &Self::Data, mcts: &TRSMCTS) -> Value {
        match *self {
            MCTSState::Terminal(th) => {
                let hh = mcts.terminals[th].trs;
                let trs = &mcts.hypotheses[hh].object.trs;
                let trs_string = trs.utrs.pretty(trs.lex.signature());
                json!({
                    "type": "terminal",
                    "trs": trs_string,
                })
            }
            MCTSState::Revision(rh) => {
                let trs_string = data.trs.utrs.pretty(data.trs.lex.signature());
                let playout_string = match mcts.revisions[rh].playout {
                    PlayoutState::Failed => "failed".to_string(),
                    PlayoutState::Untried => "untried".to_string(),
                    PlayoutState::Success(hh) => {
                        let playout = &mcts.hypotheses[hh].object.trs;
                        playout.utrs.pretty(playout.lex.signature())
                    }
                };
                json!({
                    "type": "revision",
                    "n": data.n,
                    "trs": trs_string,
                    "playout": playout_string,
                })
            }
        }
    }
    fn describe_move(
        &self,
        data: &Self::Data,
        mv: &Self::Move,
        _mcts: &TRSMCTS,
        failed: bool,
    ) -> Value {
        match *self {
            MCTSState::Terminal(_) => Value::Null,
            MCTSState::Revision(_) => {
                if failed {
                    Value::String(mv.head())
                } else {
                    Value::String(mv.pretty(&data.trs.lex))
                }
            }
        }
    }
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
                .for_each(|v| moves.push(Move::Variablize(Some(v)))),
            Some(MoveState::Compose) => self
                .trs
                .find_all_compositions()
                .into_iter()
                .for_each(|composition| moves.push(Move::Compose(Some(composition)))),
            Some(MoveState::Recurse) => self
                .trs
                .find_all_recursions()
                .into_iter()
                .for_each(|recursion| moves.push(Move::Recurse(Some(recursion)))),
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
            Move::Variablize(Some((ref m, ref tp, ref places))) => {
                let mut clauses = self.trs.utrs.clauses();
                clauses[*m] = tryo![
                    self,
                    self.trs.apply_variablization(tp, places, &clauses[*m])
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
            Move::DeleteRule(Some(n)) => {
                tryo![self, self.trs.utrs.remove_idx(n).ok()];
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
            Move::MemorizeDatum(Some(n)) => match data[n] {
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
                self.spec
                    .replace(MoveState::SampleRule(context, env, arg_tps));
            }
            Move::RegenerateRule => {
                self.path.push((mv.clone(), n));
                self.label = StateLabel::PartialRevision;
                self.spec
                    .replace(MoveState::RegenerateRule(RegenerateRuleState::Start));
            }
            Move::RegenerateThisRule(r) => {
                self.path.push((mv.clone(), n));
                self.label = StateLabel::PartialRevision;
                self.spec
                    .replace(MoveState::RegenerateRule(RegenerateRuleState::Rule(r)));
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
                            self.spec.replace(MoveState::RegenerateRule(
                                RegenerateRuleState::Place(r, ps),
                            ));
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
                                    .map(|(_, tp)| tp.apply(&env.sub))
                                    .expect("tp");
                                let arg_tps = vec![tp];
                                self.spec.replace(MoveState::RegenerateRule(
                                    RegenerateRuleState::Term(r, context, env, arg_tps),
                                ));
                            }
                            Some(p) => {
                                ps.push(p);
                                self.spec.replace(MoveState::RegenerateRule(
                                    RegenerateRuleState::Place(r, ps),
                                ));
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
                        // TODO: would be nice to avoid cloning here.
                        let new_context = tryo![
                            self,
                            context.replace(
                                &place,
                                Context::from(SituatedAtom::new(atom, self.trs.lex.signature()))
                            )
                        ];
                        if let Ok(rule) = Rule::try_from(&new_context) {
                            tryo![self, self.trs.append_clauses(vec![rule]).ok()];
                            self.n += 1;
                            self.path.push((mv.clone(), n));
                            self.spec = None;
                            self.label = StateLabel::CompleteRevision;
                        } else {
                            let tp = arg_tps[0];
                            let mut new_arg_tps = tryo![self, env.check_atom(tp, atom).ok()];
                            new_arg_tps.extend_from_slice(&arg_tps[1..]);
                            self.path.push((mv.clone(), n));
                            self.label = StateLabel::PartialRevision;
                            self.spec
                                .replace(MoveState::SampleRule(new_context, env, new_arg_tps));
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
                        let subcontext =
                            Context::from(SituatedAtom::new(atom, self.trs.lex.signature()));
                        let new_context = tryo![self, context.replace(&place, subcontext)];
                        if let Ok(rule) = Rule::try_from(&new_context) {
                            tryo![self, self.trs.utrs.remove_idx(r).ok()];
                            tryo![self, self.trs.utrs.insert_idx(r, rule).ok()];
                            self.n += 1;
                            self.path.push((mv.clone(), n));
                            self.spec = None;
                            self.label = StateLabel::CompleteRevision;
                        } else {
                            let tp = arg_tps[0];
                            let mut new_arg_tps = tryo![self, env.check_atom(tp, atom).ok()];
                            new_arg_tps.extend_from_slice(&arg_tps[1..]);
                            self.path.push((mv.clone(), n));
                            self.label = StateLabel::PartialRevision;
                            self.spec.replace(MoveState::RegenerateRule(
                                RegenerateRuleState::Term(r, new_context, env, new_arg_tps),
                            ));
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
    type MoveEval = ThompsonMoveEvaluator;
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
            best: std::f64::NEG_INFINITY,
            search_time: 0.0,
            trial_start: None,
            count: 0,
        }
    }
    pub fn start_trial(&mut self) {
        self.trial_start.replace(std::time::Instant::now());
    }
    pub fn finish_trial(&mut self) {
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
        let mut trs = state.trs.clone();
        trs.utrs.canonicalize(&mut HashMap::new());
        let (meta, prior) = state.path.clone().into_iter().fold(
            (Vec::with_capacity(state.path.len()), 0.0),
            |(mut meta, prior), (mv, n)| {
                meta.push(mv);
                (meta, prior - 2f64.ln() - (n as f64).ln())
            },
        );
        let object = MCTSObj::new(trs, time, count, meta, prior);
        let model = MCTSModel::new(self.model);
        HypothesisHandle(self.hypotheses.insert(Hypothesis::new(object, model)))
    }
    pub fn rm_hypothesis(&mut self, hh: HypothesisHandle) {
        self.hypotheses.remove(hh.0);
    }
    pub fn clear(&mut self) {
        self.hypotheses.clear();
        self.terminals.clear();
        self.revisions.clear();
        self.best = std::f64::NEG_INFINITY;
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
    //            return (None, std::f64::NEG_INFINITY);
    //        }
    //        // Take the move and update the state accordingly.
    //        match Revision::make_move_inner(&mv, &spec, &self.data, &trs) {
    //            MoveResult::Failed => {
    //                return (None, std::f64::NEG_INFINITY);
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
    //        (None, std::f64::NEG_INFINITY)
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

impl<'a, 'b> MoveEvaluator<TRSMCTS<'a, 'b>> for ThompsonMoveEvaluator {
    type NodeStatistics = QNN;
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
        unexplored_first(moves, nh, tree.mcts(), tree.tree(), thompson_sample, rng)
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
            MCTSState::Terminal(th) => mcts.hypotheses[mcts.terminals[th].trs].lposterior,
            MCTSState::Revision(rh) => match mcts.revisions[rh].playout {
                PlayoutState::Untried => panic!("shouldn't reread untried playout"),
                PlayoutState::Failed => std::f64::NEG_INFINITY,
                PlayoutState::Success(hh) => mcts.hypotheses[hh].lposterior,
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
            MCTSState::Terminal(th) => {
                mcts.hypotheses[mcts.terminals[th].trs].log_posterior(mcts.data);
                mcts.hypotheses[mcts.terminals[th].trs].lposterior
            }
            MCTSState::Revision(rh) => match mcts.revisions[rh].playout.clone() {
                PlayoutState::Untried => {
                    if let Some(mut state) = data.playout(mcts, rng) {
                        let hh = mcts.make_hypothesis(&mut state);
                        mcts.revisions[rh].playout = PlayoutState::Success(hh);
                        mcts.hypotheses[hh].log_posterior(mcts.data)
                    } else {
                        mcts.revisions[rh].playout = PlayoutState::Failed;
                        std::f64::NEG_INFINITY
                    }
                }
                // Note: the new DAG representation means we create multiple
                // nodes sharing the same state. They share a single playout,
                // which may not be exactly correct.
                PlayoutState::Failed => std::f64::NEG_INFINITY,
                PlayoutState::Success(hh) => mcts.hypotheses[hh].lposterior,
            },
        };
        if score > mcts.best {
            mcts.best = score;
        }
        score
    }
}

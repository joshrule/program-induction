use generational_arena::{Arena, Index};
use itertools::Itertools;
use mcts::{
    MoveCheck, MoveEvaluator, MoveHandle, NodeHandle, NodeStatistic, SearchTree, State,
    StateEvaluator, TreeStore, MCTS,
};
use rand::{
    distributions::Bernoulli,
    prelude::{Distribution, Rng, SliceRandom},
};
use serde_json::Value;
use std::{cmp::Ordering, collections::HashMap, convert::TryFrom};
use term_rewriting::{Atom, Context, Rule, RuleContext, SituatedAtom};
use trs::{
    Composition, Datum, Hypothesis, Lexicon, ModelParams, ProbabilisticModel, Recursion,
    Variablization, TRS,
};
use utils::logsumexp;

type RevisionHandle = usize;
type TerminalHandle = usize;
type TRSHandle = usize;
type Hyp<'ctx, 'b> = Hypothesis<MCTSObj<'ctx, 'b>, &'b Datum, MCTSModel<'ctx, 'b>>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct HypothesisHandle(Index);

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum StateHandle {
    Revision(RevisionHandle),
    Terminal(TerminalHandle),
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct MCTSState {
    handle: StateHandle,
}

#[derive(Debug, Clone)]
pub struct Revision<'ctx> {
    n: usize,
    trs: TRSHandle,
    spec: Option<MCTSMoveState<'ctx>>,
    playout: PlayoutState<'ctx, HypothesisHandle>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Terminal {
    trs: HypothesisHandle,
}

pub enum StateKind<'ctx> {
    Terminal(Terminal),
    Revision(Revision<'ctx>),
}

pub enum MoveResult<'ctx, 'b> {
    Failed,
    Terminal,
    Revision(Option<TRS<'ctx, 'b>>, Option<MCTSMoveState<'ctx>>),
}

macro_rules! r#tryo {
    ($expr:expr) => {
        match $expr {
            std::option::Option::Some(val) => val,
            std::option::Option::None => {
                return $crate::trs::mcts::MoveResult::Failed;
            }
        }
    };
}

pub struct TRSMCTS<'ctx, 'b> {
    pub lexicon: Lexicon<'ctx, 'b>,
    pub bg: &'b [Rule],
    pub deterministic: bool,
    pub lo: usize,
    pub hi: usize,
    pub data: &'b [&'b Datum],
    pub trss: Vec<TRS<'ctx, 'b>>,
    pub hypotheses: Arena<Hypothesis<MCTSObj<'ctx, 'b>, &'b Datum, MCTSModel<'ctx, 'b>>>,
    pub revisions: Vec<Revision<'ctx>>,
    pub terminals: Vec<Terminal>,
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
    pub meta: Vec<MCTSMove<'ctx>>,
    pub meta_prior: f64,
}

impl<'ctx, 'b> MCTSObj<'ctx, 'b> {
    pub fn new(
        trs: TRS<'ctx, 'b>,
        time: f64,
        count: usize,
        meta: Vec<MCTSMove<'ctx>>,
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

pub struct MCTSModel<'a, 'b> {
    params: ModelParams,
    evals: HashMap<&'b Datum, f64>,
    phantom: std::marker::PhantomData<TRS<'a, 'b>>,
}

impl<'a, 'b> MCTSModel<'a, 'b> {
    pub fn new(params: ModelParams) -> Self {
        MCTSModel {
            params,
            evals: HashMap::new(),
            phantom: std::marker::PhantomData,
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

impl<'a, 'b> ProbabilisticModel for MCTSModel<'a, 'b> {
    type Object = MCTSObj<'a, 'b>;
    type Datum = &'b Datum;
    fn log_prior(&mut self, object: &Self::Object) -> f64 {
        let lprior = object.meta_prior;
        lprior
    }
    fn single_log_likelihood<DataIter>(
        &mut self,
        object: &Self::Object,
        data: &Self::Datum,
    ) -> f64 {
        object
            .trs
            .single_log_likelihood(data, self.params.likelihood)
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
    let score = exploit + explore;
    score
}

pub fn thompson_sample<R: Rng>(parent: &[f64], child: &[f64], mcts: &TRSMCTS, rng: &mut R) -> f64 {
    let m = match mcts.params.selection {
        Selection::Thompson(m) => m,
        x => panic!("in thompson_sample but Selection is {:?}", x),
    };
    let n = child.len() as u32;
    let source = if Bernoulli::from_ratio(n, m + n).unwrap().sample(rng) {
        child
    } else {
        parent
    };
    *source.choose(rng).unwrap()
}

pub fn compute_path<'ctx, 'b>(
    parent: NodeHandle,
    mv: &MCTSMove<'ctx>,
    tree: &TreeStore<TRSMCTS<'ctx, 'b>>,
    children: bool,
) -> (Vec<MCTSMove<'ctx>>, f64) {
    let path = tree.path_tree(parent);
    let mut moves = path.iter().map(|mh| tree.mv(*mh).mov.clone()).collect_vec();
    moves.push(mv.clone());
    // Collect cost of choices so far.
    let mut score = path
        .iter()
        .map(|mh| -(tree.siblings_tree(*mh).len() as f64).ln())
        .sum();
    //println!("score {}", score);
    // Add cost of final choice.
    score -= (tree.node(parent).outgoing.len() as f64).ln();
    //println!("score {}", score);
    // Leave reward at each node.
    score -= ((moves.len() + (children as usize)) as f64) * 2f64.ln();
    //println!("score {}", score);
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
///   let place = vec![0, 1, 1, 1];
///   println!("spot: {}", context.at(&place).expect("hole").pretty(lex.signature()));
///   let fillers = rulecontext_fillers(&lex, &context, &place);
///   println!("{}", fillers.len());
///   for a in fillers {
///       println!("{}", a.display(lex.signature()));
///   }
/// })
/// ```
pub fn rulecontext_fillers<'ctx, 'b>(
    lex: &Lexicon<'ctx, 'b>,
    context: &RuleContext,
    place: &[usize],
) -> Vec<Atom> {
    match context.at(place) {
        Some(&Context::Hole) => {
            if let Ok(mut env) = lex.infer_rulecontext(context) {
                let tp = context
                    .subcontexts()
                    .iter()
                    .zip(&env.tps)
                    .find(|((_, p), _)| p.as_slice() == place)
                    .map(|(_, tp)| tp.apply(&env.sub))
                    .expect("tp");
                env.invent = place[0] == 0;
                let new_atoms = env.enumerate_atoms(tp).collect_vec();
                new_atoms
                    .into_iter()
                    .filter_map(|atom| match atom {
                        None => {
                            if place == [0] {
                                None
                            } else {
                                env.new_variable().map(Atom::Variable)
                            }
                        }
                        Some(a) => {
                            if place == [0] && a.is_variable() {
                                None
                            } else {
                                Some(a)
                            }
                        }
                    })
                    .collect()
            } else {
                vec![]
            }
        }
        _ => vec![],
    }
}

pub struct BestSoFarMoveEvaluator;
pub struct RescaledByBestMoveEvaluator;
pub struct RelativeMassMoveEvaluator;
pub struct ThompsonMoveEvaluator;

pub struct MCTSStateEvaluator;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum MCTSMoveState<'ctx> {
    SampleRule(RuleContext),
    RegenerateRule(Option<(usize, RuleContext)>),
    Compose(Vec<Composition<'ctx>>),
    Recurse(Vec<Recursion<'ctx>>),
    Variablize(Vec<Variablization<'ctx>>),
    MemorizeDatum,
    DeleteRule,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MCTSMove<'ctx> {
    SampleRule,
    SampleAtom(Atom),
    RegenerateRule,
    RegenerateThisRule(usize, RuleContext),
    MemorizeAll,
    MemorizeDatum(Option<usize>),
    DeleteRule(Option<usize>),
    Variablize(Option<Variablization<'ctx>>),
    Compose(Option<Composition<'ctx>>),
    Recurse(Option<Recursion<'ctx>>),
    Generalize,
    AntiUnify,
    Stop,
}

#[derive(Debug, Clone)]
pub enum PlayoutState<'ctx, T: std::fmt::Debug + Copy> {
    Untried(Vec<MCTSMove<'ctx>>, f64),
    Failed,
    Success(T),
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

impl<'ctx> MCTSMove<'ctx> {
    fn head(&self) -> String {
        match *self {
            MCTSMove::MemorizeAll => "MemorizeAll".to_string(),
            MCTSMove::SampleAtom(_) => "SampleAtom".to_string(),
            MCTSMove::RegenerateThisRule(..) => "RegenerateThisRule".to_string(),
            MCTSMove::Variablize(_) => "Variablize".to_string(),
            MCTSMove::DeleteRule(_) => "DeleteRule".to_string(),
            MCTSMove::MemorizeDatum(_) => "MemorizeDatum".to_string(),
            MCTSMove::SampleRule => "SampleRule".to_string(),
            MCTSMove::RegenerateRule => "RegenerateRule".to_string(),
            MCTSMove::Generalize => "Generalize".to_string(),
            MCTSMove::AntiUnify => "AntiUnify".to_string(),
            MCTSMove::Compose(_) => "Compose".to_string(),
            MCTSMove::Recurse(_) => "Recurse".to_string(),
            MCTSMove::Stop => "Stop".to_string(),
        }
    }
    fn pretty(&self, lex: &Lexicon) -> String {
        match *self {
            MCTSMove::MemorizeAll => format!("MemorizeAll"),
            MCTSMove::MemorizeDatum(Some(n)) => format!("MemorizeDatum({})", n),
            MCTSMove::SampleAtom(atom) => format!("SampleAtom({})", atom.display(lex.signature())),
            MCTSMove::RegenerateThisRule(n, ref c) => {
                format!("RegenerateThisRule({}, {})", n, c.pretty(lex.signature()))
            }
            MCTSMove::DeleteRule(Some(n)) => format!("DeleteRule({})", n),
            MCTSMove::Variablize(Some((ref n, ref t, ref ps))) => {
                format!("Variablize({}, {}, {:?})", n, t, ps)
            }
            MCTSMove::Compose(Some((ref t, ref p1, ref p2, ref tp))) => format!(
                "Compose({}, {:?}, {:?}, {})",
                t.pretty(lex.signature()),
                p1,
                p2,
                tp
            ),
            MCTSMove::Recurse(Some((ref t, ref p1, ref p2, ref tp))) => format!(
                "Recurse({}, {:?}, {:?}, {})",
                t.pretty(lex.signature()),
                p1,
                p2,
                tp
            ),
            MCTSMove::Variablize(None) => "Variablize".to_string(),
            MCTSMove::DeleteRule(None) => "DeleteRule".to_string(),
            MCTSMove::MemorizeDatum(None) => "MemorizeDatum".to_string(),
            MCTSMove::SampleRule => "SampleRule".to_string(),
            MCTSMove::RegenerateRule => "RegenerateRule".to_string(),
            MCTSMove::Generalize => "Generalize".to_string(),
            MCTSMove::AntiUnify => "AntiUnify".to_string(),
            MCTSMove::Compose(None) => "Compose".to_string(),
            MCTSMove::Recurse(None) => "Recurse".to_string(),
            MCTSMove::Stop => "Stop".to_string(),
        }
    }
}
impl<'ctx> std::fmt::Display for MCTSMove<'ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            MCTSMove::MemorizeAll => write!(f, "MemorizeAll"),
            MCTSMove::MemorizeDatum(Some(n)) => write!(f, "MemorizeDatum({})", n),
            MCTSMove::Variablize(Some((ref n, ref t, ref ps))) => {
                write!(f, "Variablize({}, {}, {:?})", n, t, ps)
            }
            MCTSMove::SampleAtom(atom) => write!(f, "SampleAtom({:?})", atom),
            MCTSMove::RegenerateThisRule(n, _) => write!(f, "RegenerateThisRule({}, context)", n),
            MCTSMove::DeleteRule(Some(n)) => write!(f, "DeleteRule({})", n),
            MCTSMove::MemorizeDatum(None) => write!(f, "MemorizeDatum"),
            MCTSMove::SampleRule => write!(f, "SampleRule"),
            MCTSMove::RegenerateRule => write!(f, "RegenerateRule"),
            MCTSMove::DeleteRule(None) => write!(f, "DeleteRule"),
            MCTSMove::Generalize => write!(f, "Generalize"),
            MCTSMove::AntiUnify => write!(f, "AntiUnify"),
            MCTSMove::Variablize(None) => write!(f, "Variablize"),
            MCTSMove::Compose(None) => write!(f, "Compose"),
            MCTSMove::Recurse(None) => write!(f, "Recurse"),
            MCTSMove::Compose(_) => write!(f, "Compose(_)"),
            MCTSMove::Recurse(_) => write!(f, "Recurse(_)"),
            MCTSMove::Stop => write!(f, "Stop"),
        }
    }
}

impl Terminal {
    pub fn new(trs: HypothesisHandle) -> Self {
        Terminal { trs }
    }
}

impl<'ctx> Revision<'ctx> {
    pub fn new(
        trs: TRSHandle,
        spec: Option<MCTSMoveState<'ctx>>,
        n: usize,
        (path, prior): (Vec<MCTSMove<'ctx>>, f64),
    ) -> Self {
        Revision {
            trs,
            spec,
            n,
            playout: PlayoutState::Untried(path, prior),
        }
    }
    pub fn show(&self) {
        println!("n: {}", self.n);
        println!("trs: {}", self.trs);
        println!("playout: {:?}", self.playout);
        println!("spec: {:?}", self.spec);
    }
    fn available_moves_inner<'b>(
        trs: &TRS<'ctx, 'b>,
        spec: &Option<MCTSMoveState<'ctx>>,
        n: usize,
        mcts: &TRSMCTS<'ctx, 'b>,
    ) -> Vec<MCTSMove<'ctx>> {
        let mut moves = vec![];
        match spec {
            None => {
                // Search can always stop.
                moves.push(MCTSMove::Stop);
                if n < mcts.params.max_revisions {
                    // Search can always sample a new rule.
                    moves.push(MCTSMove::SampleRule);
                    // A TRS must have a rule in order to regenerate or generalize.
                    if !trs.is_empty() {
                        moves.push(MCTSMove::RegenerateRule);
                        moves.push(MCTSMove::Generalize);
                        moves.push(MCTSMove::Compose(None));
                        moves.push(MCTSMove::Recurse(None));
                        moves.push(MCTSMove::Variablize(None));
                    }
                    // A TRS must have >1 rule to delete without creating cycles.
                    // Anti-unification relies on having two rules to unify.
                    if trs.len() > 1 {
                        moves.push(MCTSMove::DeleteRule(None));
                        moves.push(MCTSMove::AntiUnify);
                    }
                    // We can only add data if there's data to add.
                    if mcts.data.iter().any(|datum| match datum {
                        Datum::Partial(_) => false,
                        Datum::Full(rule) => trs.utrs.get_clause(rule).is_none(),
                    }) {
                        moves.push(MCTSMove::MemorizeAll);
                        moves.push(MCTSMove::MemorizeDatum(None));
                    }
                }
            }
            Some(MCTSMoveState::Variablize(ref vs)) => vs
                .iter()
                .cloned()
                .for_each(|v| moves.push(MCTSMove::Variablize(Some(v)))),
            Some(MCTSMoveState::Compose(ref compositions)) => compositions
                .iter()
                .cloned()
                .for_each(|composition| moves.push(MCTSMove::Compose(Some(composition)))),
            Some(MCTSMoveState::Recurse(ref recursions)) => recursions
                .iter()
                .cloned()
                .for_each(|recursion| moves.push(MCTSMove::Recurse(Some(recursion)))),
            Some(MCTSMoveState::MemorizeDatum) => {
                (0..mcts.data.len())
                    .filter(|idx| match mcts.data[*idx] {
                        Datum::Partial(_) => false,
                        Datum::Full(ref rule) => trs.utrs.get_clause(rule).is_none(),
                    })
                    .for_each(|idx| moves.push(MCTSMove::MemorizeDatum(Some(idx))));
            }
            Some(MCTSMoveState::DeleteRule) => {
                (0..trs.len()).for_each(|idx| moves.push(MCTSMove::DeleteRule(Some(idx))));
            }
            Some(MCTSMoveState::SampleRule(ref context))
            | Some(MCTSMoveState::RegenerateRule(Some((_, ref context)))) => {
                if let Some(place) = context.leftmost_hole() {
                    rulecontext_fillers(&trs.lex, &context, &place)
                        .into_iter()
                        .map(MCTSMove::SampleAtom)
                        .for_each(|mv| moves.push(mv));
                }
            }
            Some(MCTSMoveState::RegenerateRule(None)) => {
                for (i, rule) in trs.utrs.rules.iter().enumerate() {
                    let rulecontext = RuleContext::from(rule.clone());
                    for (_, place) in rulecontext.subcontexts() {
                        let mut context = rulecontext.replace(&place, Context::Hole).unwrap();
                        context.canonicalize(&mut HashMap::new());
                        if RuleContext::is_valid(&context.lhs, &context.rhs) {
                            moves.push(MCTSMove::RegenerateThisRule(i, context));
                        }
                    }
                }
            }
        }
        moves
    }
    pub fn available_moves<'b>(&self, mcts: &TRSMCTS<'ctx, 'b>) -> Vec<MCTSMove<'ctx>> {
        Revision::available_moves_inner(&mcts.trss[self.trs], &self.spec, self.n, mcts)
    }
    pub fn make_move_inner<'b>(
        mv: &MCTSMove<'ctx>,
        spec: &Option<MCTSMoveState<'ctx>>,
        data: &[&'b Datum],
        trs: &TRS<'ctx, 'b>,
    ) -> MoveResult<'ctx, 'b> {
        match *mv {
            MCTSMove::Stop => MoveResult::Terminal,
            MCTSMove::Generalize => {
                let trs = tryo![trs.generalize().ok()];
                MoveResult::Revision(Some(trs), None)
            }
            MCTSMove::AntiUnify => {
                let trs = tryo![trs.lgg().ok()];
                MoveResult::Revision(Some(trs), None)
            }
            MCTSMove::Compose(None) => {
                let compositions = trs.find_all_compositions();
                if compositions.is_empty() {
                    MoveResult::Failed
                } else {
                    MoveResult::Revision(None, Some(MCTSMoveState::Compose(compositions)))
                }
            }
            MCTSMove::Compose(Some(ref composition)) => {
                let trs = tryo![trs.compose_by(composition)];
                MoveResult::Revision(Some(trs), None)
            }
            MCTSMove::Recurse(None) => {
                let recursions = trs.find_all_recursions();
                if recursions.is_empty() {
                    MoveResult::Failed
                } else {
                    MoveResult::Revision(None, Some(MCTSMoveState::Recurse(recursions)))
                }
            }
            MCTSMove::Recurse(Some(ref recursion)) => {
                let trs = tryo![trs.recurse_by(recursion)];
                MoveResult::Revision(Some(trs), None)
            }
            MCTSMove::Variablize(None) => {
                let vs = trs.find_all_variablizations();
                if vs.is_empty() {
                    MoveResult::Failed
                } else {
                    MoveResult::Revision(None, Some(MCTSMoveState::Variablize(vs)))
                }
            }
            MCTSMove::Variablize(Some((ref m, ref tp, ref places))) => {
                let mut clauses = trs.utrs.clauses();
                clauses[*m] = tryo![trs.apply_variablization(tp, places, &clauses[*m])];
                let new_trs = tryo![trs.adopt_rules(&mut clauses.clone())];
                MoveResult::Revision(Some(new_trs), None)
            }
            MCTSMove::DeleteRule(None) => {
                MoveResult::Revision(None, Some(MCTSMoveState::DeleteRule))
            }
            MCTSMove::DeleteRule(Some(n)) => {
                let mut trs = trs.clone();
                tryo![trs.utrs.remove_idx(n).ok()];
                MoveResult::Revision(Some(trs), None)
            }
            MCTSMove::MemorizeAll => {
                let mut trs = trs.clone();
                let new_data = data
                    .iter()
                    .filter_map(|d| match d {
                        Datum::Partial(_) => None,
                        Datum::Full(rule) => {
                            if trs.utrs.get_clause(rule).is_none() {
                                Some(rule.clone())
                            } else {
                                None
                            }
                        }
                    })
                    .collect_vec();
                if new_data.is_empty() {
                    return MoveResult::Failed;
                }
                tryo![trs.append_clauses(new_data).ok()];
                return MoveResult::Revision(Some(trs), None);
            }
            MCTSMove::MemorizeDatum(None) => {
                MoveResult::Revision(None, Some(MCTSMoveState::MemorizeDatum))
            }
            MCTSMove::MemorizeDatum(Some(n)) => {
                let mut trs = trs.clone();
                match data[n] {
                    Datum::Partial(_) => panic!("can't memorize partial data"),
                    Datum::Full(ref rule) => tryo![trs.append_clauses(vec![rule.clone()]).ok()],
                }
                MoveResult::Revision(Some(trs), None)
            }
            MCTSMove::SampleRule => {
                let spec = Some(MCTSMoveState::SampleRule(RuleContext::default()));
                MoveResult::Revision(None, spec)
            }
            MCTSMove::RegenerateRule => {
                let spec = Some(MCTSMoveState::RegenerateRule(None));
                MoveResult::Revision(None, spec)
            }
            MCTSMove::RegenerateThisRule(n, ref context) => {
                let spec = Some(MCTSMoveState::RegenerateRule(Some((n, context.clone()))));
                MoveResult::Revision(None, spec)
            }
            MCTSMove::SampleAtom(atom) => match spec {
                Some(MCTSMoveState::SampleRule(ref context)) => {
                    let place = tryo![context.leftmost_hole()];
                    let new_context = tryo![context.replace(
                        &place,
                        Context::from(SituatedAtom::new(atom, trs.lex.signature()))
                    )];
                    if let Ok(rule) = Rule::try_from(&new_context) {
                        let mut trs = trs.clone();
                        tryo![trs.append_clauses(vec![rule]).ok()];
                        MoveResult::Revision(Some(trs), None)
                    } else {
                        MoveResult::Revision(None, Some(MCTSMoveState::SampleRule(new_context)))
                    }
                }
                Some(MCTSMoveState::RegenerateRule(Some((n, ref context)))) => {
                    let place = tryo![context.leftmost_hole()];
                    let new_context = tryo![context.replace(
                        &place,
                        Context::from(SituatedAtom::new(atom, trs.lex.signature()))
                    )];
                    if let Ok(rule) = Rule::try_from(&new_context) {
                        let mut trs = trs.clone();
                        tryo![trs.utrs.remove_idx(*n).ok()];
                        tryo![trs.utrs.insert_idx(*n, rule).ok()];
                        MoveResult::Revision(Some(trs), None)
                    } else {
                        let spec = Some(MCTSMoveState::RegenerateRule(Some((*n, new_context))));
                        MoveResult::Revision(None, spec)
                    }
                }
                _ => panic!("MCTSMoveState doesn't match MCTSMove"),
            },
        }
    }
    pub fn make_move<'b>(
        parent: NodeHandle,
        mv: &MCTSMove<'ctx>,
        mcts: &mut TRSMCTS<'ctx, 'b>,
        tree: &TreeStore<TRSMCTS<'ctx, 'b>>,
        handle: RevisionHandle,
    ) -> Option<MCTSState> {
        let trsh = mcts.revisions[handle].trs;
        let n = mcts.revisions[handle].n;
        let state = match Revision::make_move_inner(
            mv,
            &mcts.revisions[handle].spec,
            mcts.data,
            &mcts.trss[trsh],
        ) {
            MoveResult::Failed => None,
            MoveResult::Terminal => {
                let path = compute_path(parent, mv, tree, false);
                let hh = mcts.make_hypothesis(mcts.trss[trsh].clone(), path);
                Some(StateKind::Terminal(Terminal::new(hh)))
            }
            MoveResult::Revision(None, new_spec) => {
                let new_n = n + (new_spec.is_none() as usize);
                let children =
                    !Revision::available_moves_inner(&mcts.trss[trsh], &new_spec, new_n, mcts)
                        .is_empty();
                let path = compute_path(parent, mv, tree, children);
                Some(StateKind::Revision(Revision::new(
                    trsh, new_spec, new_n, path,
                )))
            }
            MoveResult::Revision(Some(trs), new_spec) => {
                let new_n = n + (new_spec.is_none() as usize);
                let children =
                    !Revision::available_moves_inner(&trs, &new_spec, new_n, mcts).is_empty();
                let path = compute_path(parent, mv, tree, children);
                let new_trsh = mcts.find_trs(trs);
                Some(StateKind::Revision(Revision::new(
                    new_trsh, new_spec, new_n, path,
                )))
            }
        }?;
        Some(mcts.add_state(state))
    }
    pub fn playout<'b, R: Rng>(
        rh: RevisionHandle,
        mcts: &mut TRSMCTS<'ctx, 'b>,
        start_path: &[MCTSMove<'ctx>],
        start_prior: f64,
        rng: &mut R,
    ) -> Option<(TRS<'ctx, 'b>, Vec<MCTSMove<'ctx>>, f64)> {
        let trsh = mcts.revisions[rh].trs;
        // TODO: Maybe try backtracking instead of a fixed count?
        for _ in 0..10 {
            let mut trs = mcts.trss[trsh].clone();
            let mut spec = mcts.revisions[rh].spec.clone();
            let mut n = mcts.revisions[rh].n;
            let mut progress = true;
            let mut depth = start_path.len();
            let mut path = start_path.to_vec();
            let mut prior = start_prior;
            while progress && depth < mcts.params.max_depth {
                progress = false;
                // Compute the available moves.
                let mut moves = Revision::available_moves_inner(&trs, &spec, n, mcts);
                // Choose a move (random policy).
                let moves_len = moves.len();
                moves.shuffle(rng);
                if let Some(idx) = moves.iter().position(|mv| *mv == MCTSMove::Stop) {
                    moves.swap(0, idx);
                }
                for mv in moves {
                    match Revision::make_move_inner(&mv, &spec, &mcts.data, &trs) {
                        MoveResult::Failed => {
                            continue;
                        }
                        MoveResult::Terminal => {
                            path.push(mv.clone());
                            prior -= (moves_len as f64).ln();
                            return Some((trs, path, prior));
                        }
                        MoveResult::Revision(new_trs, new_spec) => {
                            path.push(mv.clone());
                            prior -= (moves_len as f64).ln() + 2f64.ln();
                            n += new_spec.is_none() as usize;
                            if let Some(new_trs) = new_trs {
                                trs = new_trs;
                            }
                            depth += 1;
                            spec = new_spec;
                            progress = true;
                            break;
                        }
                    }
                }
            }
        }
        None
    }
}
impl<'ctx> Eq for Revision<'ctx> {}
impl<'ctx> PartialEq for Revision<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        self.trs == other.trs && self.spec == other.spec
    }
}

impl<'ctx, 'b> State<TRSMCTS<'ctx, 'b>> for MCTSState {
    type Move = MCTSMove<'ctx>;
    type MoveList = Vec<Self::Move>;
    fn available_moves(&self, mcts: &TRSMCTS<'ctx, 'b>) -> Self::MoveList {
        match self.handle {
            StateHandle::Terminal(..) => vec![],
            StateHandle::Revision(rh) => mcts.revisions[rh].available_moves(mcts),
        }
    }
    fn make_move(
        &self,
        parent: NodeHandle,
        mv: &Self::Move,
        mcts: &mut TRSMCTS<'ctx, 'b>,
        tree: &TreeStore<TRSMCTS<'ctx, 'b>>,
    ) -> Option<Self> {
        match self.handle {
            StateHandle::Terminal(..) => panic!("cannot move from terminal"),
            StateHandle::Revision(rh) => Revision::make_move(parent, mv, mcts, tree, rh),
        }
    }
    fn check_move(
        mh: MoveHandle,
        mcts: &mut TRSMCTS<'ctx, 'b>,
        tree: &TreeStore<TRSMCTS<'ctx, 'b>>,
    ) -> MoveCheck<TRSMCTS<'ctx, 'b>> {
        let ph = tree.mv(mh).parent;
        let ch;
        match tree.mv(mh).child {
            Some(child) => ch = child,
            None => return MoveCheck::Expected,
        };
        match tree.node(ph).state.handle {
            StateHandle::Terminal(..) => panic!("cannot move from terminal"),
            StateHandle::Revision(rh) => {
                let n = mcts.revisions[rh].n;
                let spec = &mcts.revisions[rh].spec;
                match Revision::make_move_inner(
                    &tree.mv(mh).mov,
                    spec,
                    mcts.data,
                    &mcts.trss[mcts.revisions[rh].trs],
                ) {
                    MoveResult::Failed => MoveCheck::Failed,
                    MoveResult::Terminal => match tree.node(ch).state.handle {
                        StateHandle::Revision(_) => {
                            let path = compute_path(ph, &tree.mv(mh).mov, tree, false);
                            let trs = mcts.trss[mcts.revisions[rh].trs].clone();
                            let hh = mcts.make_hypothesis(trs, path);
                            MoveCheck::NewState(mcts.add_terminal(Terminal::new(hh)))
                        }
                        StateHandle::Terminal(th) => {
                            let old_trs = &mcts.hypotheses[mcts.terminals[th].trs].object.trs;
                            if TRS::same_shape(old_trs, &mcts.trss[mcts.revisions[rh].trs]) {
                                MoveCheck::Expected
                            } else {
                                let path = compute_path(ph, &tree.mv(mh).mov, tree, false);
                                let trs = mcts.trss[mcts.revisions[rh].trs].clone();
                                let hh = mcts.make_hypothesis(trs, path);
                                MoveCheck::NewState(mcts.add_terminal(Terminal::new(hh)))
                            }
                        }
                    },
                    MoveResult::Revision(maybe_trs, new_spec) => {
                        let new_trs = match maybe_trs {
                            Some(trs) => trs,
                            None => mcts.trss[mcts.revisions[rh].trs].clone(),
                        };
                        let new_n = n + (new_spec.is_none() as usize);
                        let children =
                            !Revision::available_moves_inner(&new_trs, &new_spec, new_n, mcts)
                                .is_empty();
                        match tree.node(ch).state.handle {
                            StateHandle::Revision(orh) => {
                                let old_trs = &mcts.trss[mcts.revisions[orh].trs];
                                if TRS::same_shape(old_trs, &new_trs)
                                    && mcts.revisions[orh].n == new_n
                                    && mcts.revisions[orh].spec == new_spec
                                {
                                    let (path, prior) =
                                        compute_path(ph, &tree.mv(mh).mov, tree, children);
                                    if mcts.update_playout(orh) {
                                        let new_trsh = mcts.find_trs(new_trs);
                                        MoveCheck::NewState(mcts.add_revision(Revision::new(
                                            new_trsh,
                                            new_spec,
                                            new_n,
                                            (path, prior),
                                        )))
                                    } else {
                                        MoveCheck::Expected
                                    }
                                } else {
                                    let path = compute_path(ph, &tree.mv(mh).mov, tree, children);
                                    let new_trsh = mcts.find_trs(new_trs);
                                    let new_n = n + (new_spec.is_none() as usize);
                                    MoveCheck::NewState(mcts.add_revision(Revision::new(
                                        new_trsh, new_spec, new_n, path,
                                    )))
                                }
                            }
                            StateHandle::Terminal(_) => {
                                let path = compute_path(ph, &tree.mv(mh).mov, tree, children);
                                let new_trsh = mcts.find_trs(new_trs.clone());
                                MoveCheck::NewState(
                                    mcts.add_revision(Revision::new(
                                        new_trsh, new_spec, new_n, path,
                                    )),
                                )
                            }
                        }
                    }
                }
            }
        }
    }
    fn add_moves_for_new_data(
        &self,
        moves: &[Self::Move],
        mcts: &mut TRSMCTS<'ctx, 'b>,
    ) -> Vec<Self::Move> {
        self.available_moves(mcts)
            .into_iter()
            .filter(|m| !moves.contains(&m))
            .collect()
    }
    fn describe_move(&self, mv: &Self::Move, mcts: &TRSMCTS, failed: bool) -> Value {
        match self.handle {
            StateHandle::Terminal(_) => Value::Null,
            StateHandle::Revision(rh) => {
                if failed {
                    Value::String(mv.head())
                } else {
                    Value::String(mv.pretty(&mcts.trss[mcts.revisions[rh].trs].lex))
                }
            }
        }
    }
    fn describe_self(&self, mcts: &TRSMCTS) -> Value {
        match self.handle {
            StateHandle::Terminal(th) => {
                let hh = mcts.terminals[th].trs;
                let trs = &mcts.hypotheses[hh].object.trs;
                let trs_string = trs.utrs.pretty(trs.lex.signature());
                json!({
                    "type": "terminal",
                    "trs": trs_string,
                })
            }
            StateHandle::Revision(rh) => {
                let hh = mcts.revisions[rh].trs;
                let trs = &mcts.trss[hh];
                let trs_string = trs.utrs.pretty(trs.lex.signature());
                let playout_string = match mcts.revisions[rh].playout {
                    PlayoutState::Failed => "failed".to_string(),
                    PlayoutState::Untried(..) => "untried".to_string(),
                    PlayoutState::Success(hh) => {
                        let playout = &mcts.hypotheses[hh].object.trs;
                        playout.utrs.pretty(playout.lex.signature())
                    }
                };
                json!({
                    "type": "revision",
                    "n": mcts.revisions[rh].n,
                    "trs": trs_string,
                    "playout": playout_string,
                })
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
            trss: vec![],
            hypotheses: Arena::new(),
            terminals: vec![],
            revisions: vec![],
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
    pub fn add_state(&mut self, state: StateKind<'ctx>) -> MCTSState {
        match state {
            StateKind::Terminal(h) => self.add_terminal(h),
            StateKind::Revision(r) => self.add_revision(r),
        }
    }
    pub fn add_revision(&mut self, state: Revision<'ctx>) -> MCTSState {
        self.revisions.push(state);
        let handle = StateHandle::Revision(self.revisions.len() - 1);
        MCTSState { handle }
    }
    pub fn add_terminal(&mut self, state: Terminal) -> MCTSState {
        let th = match self.terminals.iter().position(|t| *t == state) {
            Some(th) => th,
            None => {
                self.terminals.push(state);
                self.terminals.len() - 1
            }
        };
        let handle = StateHandle::Terminal(th);
        MCTSState { handle }
    }
    pub fn find_trs(&mut self, mut trs: TRS<'ctx, 'b>) -> TRSHandle {
        trs.utrs.canonicalize(&mut HashMap::new());
        match self.trss.iter().position(|t| TRS::same_shape(&t, &trs)) {
            Some(trsh) => trsh,
            None => {
                self.trss.push(trs);
                self.trss.len() - 1
            }
        }
    }
    pub fn make_hypothesis(
        &mut self,
        mut trs: TRS<'ctx, 'b>,
        (meta, prior): (Vec<MCTSMove<'ctx>>, f64),
    ) -> HypothesisHandle {
        let time = self.search_time
            + self
                .trial_start
                .map(|ts| ts.elapsed().as_secs_f64())
                .unwrap_or(0.0);
        let count = self.count;
        self.count += 1;
        trs.utrs.canonicalize(&mut HashMap::new());
        let object = MCTSObj::new(trs, time, count, meta, prior);
        let model = MCTSModel::new(self.model);
        HypothesisHandle(self.hypotheses.insert(Hypothesis::new(object, model)))
    }
    pub fn root(&mut self) -> MCTSState {
        let mut trs = TRS::new_unchecked(&self.lexicon, self.deterministic, self.bg, vec![]);
        trs.utrs.lo = self.lo;
        trs.utrs.hi = self.hi;
        trs.identify_symbols();
        let state = Revision {
            trs: self.find_trs(trs),
            spec: None,
            n: 0,
            playout: PlayoutState::Untried(vec![], 0.5f64.ln()),
        };
        self.add_revision(state)
    }
    pub fn update_hypotheses(&mut self) {
        let mut hypotheses = std::mem::replace(&mut self.hypotheses, Arena::new());
        for (_, hypothesis) in hypotheses.iter_mut() {
            let (new_trs, new_prior) = self.p_path(&hypothesis.object.meta);
            hypothesis.object.meta_prior = new_prior;
            if let Some(trs) = new_trs {
                hypothesis.object.trs = trs;
            }
            hypothesis.log_posterior(self.data);
        }
        self.hypotheses = hypotheses;
    }
    pub fn p_path(&self, moves: &[MCTSMove<'ctx>]) -> (Option<TRS<'ctx, 'b>>, f64) {
        // Set the root state.
        let mut trs = self.trss[self.revisions[0].trs].clone();
        let mut spec = None;
        let mut n = 0;
        let mut ps = Vec::with_capacity(moves.len());
        for mv in moves {
            // Ensure our move is available.
            let available_moves = Revision::available_moves_inner(&trs, &spec, n, self);
            if !available_moves.contains(mv) {
                return (None, std::f64::NEG_INFINITY);
            }
            // Take the move and update the state accordingly.
            match Revision::make_move_inner(&mv, &spec, &self.data, &trs) {
                MoveResult::Failed => {
                    return (None, std::f64::NEG_INFINITY);
                }
                MoveResult::Terminal => {
                    ps.push(available_moves.len() as f64);
                    break;
                }
                MoveResult::Revision(new_trs, new_spec) => {
                    ps.push(available_moves.len() as f64);
                    n += new_spec.is_none() as usize;
                    spec = new_spec;
                    if let Some(new_trs) = new_trs {
                        trs = new_trs;
                    }
                }
            }
        }
        if ps.len() == moves.len() {
            let prior = ps.into_iter().map(|p| -(2.0 * p).ln()).sum();
            (Some(trs), prior)
        } else {
            (None, std::f64::NEG_INFINITY)
        }
    }
    fn update_playout(&mut self, rh: RevisionHandle) -> bool {
        match self.revisions[rh].playout {
            PlayoutState::Untried(..) => false,
            PlayoutState::Failed => true,
            PlayoutState::Success(hh) => {
                let (new_trs, new_prior) = self.p_path(&self.hypotheses[hh].object.meta);
                if let Some(trs) = new_trs {
                    self.hypotheses[hh].object.meta_prior = new_prior;
                    self.hypotheses[hh].object.trs = trs;
                    false
                } else {
                    true
                }
            }
        }
    }
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
    F: Fn(&[f64], &[f64], &TRSMCTS, &mut R) -> f64,
{
    // Split the moves into those with and without children.
    let (childful, childless): (Vec<_>, Vec<_>) =
        moves.partition(|mh| tree.mv(*mh).child.is_some());
    // Take the first childless move, or perform UCT on childed moves.
    if let Some(mh) = childless
        .iter()
        .find(|mh| tree.mv(**mh).mov == MCTSMove::Stop)
    {
        Some(*mh)
    } else if let Some(mh) = childless.choose(rng) {
        Some(*mh)
    } else {
        childful
            .into_iter()
            .map(|mh| {
                let ch = tree.mv(mh).child.expect("INVARIANT: partition failed us");
                let parent = &tree.node(nh).stats;
                let child = &tree.node(ch).stats;
                let score = selector(parent, child, mcts, rng);
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
    type NodeStatistics = Vec<f64>;
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
        match state.handle {
            StateHandle::Terminal(th) => mcts.hypotheses[mcts.terminals[th].trs].lposterior,
            StateHandle::Revision(rh) => match mcts.revisions[rh].playout {
                PlayoutState::Untried(..) => panic!("shouldn't reread untried playout"),
                PlayoutState::Failed => std::f64::NEG_INFINITY,
                PlayoutState::Success(hh) => mcts.hypotheses[hh].lposterior,
            },
        }
    }
    fn evaluate<R: Rng>(
        &self,
        state: &<TRSMCTS<'a, 'b> as MCTS>::State,
        mcts: &mut TRSMCTS<'a, 'b>,
        rng: &mut R,
    ) -> Self::StateEvaluation {
        let score = match state.handle {
            StateHandle::Terminal(th) => {
                mcts.hypotheses[mcts.terminals[th].trs].log_posterior(mcts.data);
                mcts.hypotheses[mcts.terminals[th].trs].lposterior
            }
            StateHandle::Revision(rh) => match mcts.revisions[rh].playout.clone() {
                PlayoutState::Untried(path, prior) => {
                    if let Some((trs, path, prior)) = Revision::playout(rh, mcts, &path, prior, rng)
                    {
                        let hh = mcts.make_hypothesis(trs, (path, prior));
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

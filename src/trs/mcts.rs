use itertools::Itertools;
use mcts::{
    MoveEvaluator, MoveInfo, NodeHandle, NodeStatistic, SearchTree, State, StateEvaluator, Stats,
    TreeStore, MCTS,
};
use rand::{
    distributions::{Bernoulli, WeightedIndex},
    prelude::{Distribution, Rng, SliceRandom},
};
use serde_json::Value;
use std::{cmp::Ordering, collections::HashMap, convert::TryFrom};
use term_rewriting::{Atom, Context, Rule, RuleContext};
use trs::{
    Composition, Datum, Hypothesis, Lexicon, ModelParams, ProbabilisticModel, Recursion,
    Variablization, TRS,
};
use utils::{exp_normalize, logsumexp};

type RevisionHandle = usize;
type TerminalHandle = usize;
type HypothesisHandle = usize;
type TRSHandle = usize;

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
pub struct Revision {
    n: usize,
    trs: TRSHandle,
    spec: Option<MCTSMoveState>,
    playout: PlayoutState<HypothesisHandle>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Terminal {
    trs: HypothesisHandle,
}

pub enum StateKind {
    Terminal(Terminal),
    Revision(Revision),
}

pub enum MoveResult<'a, 'b> {
    Failed,
    Terminal,
    Revision(Option<TRS<'a, 'b>>, Option<MCTSMoveState>),
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

pub struct TRSMCTS<'a, 'b> {
    pub lexicon: Lexicon<'b>,
    pub bg: &'a [Rule],
    pub deterministic: bool,
    pub data: &'a [Datum],
    pub trss: Vec<TRS<'a, 'b>>,
    pub hypotheses: Vec<Hypothesis<MCTSObj<'a, 'b>, Datum, MCTSModel<'a, 'b>>>,
    pub revisions: Vec<Revision>,
    pub terminals: Vec<Terminal>,
    pub model: ModelParams,
    pub params: MCTSParams,
    pub best: f64,
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

pub struct MCTSObj<'a, 'b> {
    pub trs: TRS<'a, 'b>,
    pub metas: HashMap<Vec<MCTSMove>, f64>,
}

impl<'a, 'b> MCTSObj<'a, 'b> {
    pub fn new(trs: TRS<'a, 'b>, metas: HashMap<Vec<MCTSMove>, f64>) -> Self {
        MCTSObj { trs, metas }
    }
}

pub struct MCTSModel<'a, 'b> {
    params: ModelParams,
    evals: HashMap<Datum, f64>,
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

impl<'a, 'b> ProbabilisticModel for MCTSModel<'a, 'b> {
    type Object = MCTSObj<'a, 'b>;
    type Datum = Datum;
    fn log_prior(&mut self, object: &Self::Object) -> f64 {
        println!("metas: {:?}", object.metas);
        let lprior = logsumexp(&object.metas.values().map(|x| *x).collect_vec());
        println!("lprior: {}", lprior);
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
    fn log_likelihood(&mut self, object: &Self::Object, data: &[Datum]) -> f64 {
        object
            .trs
            .log_likelihood(data, &mut self.evals, self.params.likelihood)
    }
    fn log_posterior(&mut self, object: &Self::Object, data: &[Datum]) -> (f64, f64, f64) {
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
    println!(
        "{} + {} = {} ({} {})",
        exploit, explore, score, child.q, best,
    );
    score
}

pub fn thompson_sample<R: Rng>(
    parent: &Vec<f64>,
    child: &Vec<f64>,
    mcts: &TRSMCTS,
    rng: &mut R,
) -> f64 {
    let m = match mcts.params.selection {
        Selection::Thompson(m) => m,
        x => panic!("in thompson_sample but Selection is {:?}", x),
    };
    let n = child.len() as u32;
    println!("child: {:?}", child);
    println!("parent: {:?}", parent);
    let source = if Bernoulli::from_ratio(n, m + n).unwrap().sample(rng) {
        child
    } else {
        parent
    };
    exp_normalize(source, Some(-1000f64.ln()))
        .map(|ps| {
            println!("ps: {:?}", ps);
            source[WeightedIndex::new(ps).unwrap().sample(rng)]
        })
        .unwrap_or_else(|| *source.choose(rng).unwrap())
}

pub fn compute_paths(
    parent: NodeHandle,
    mv: &MCTSMove,
    tree: &TreeStore<TRSMCTS>,
    children: bool,
) -> HashMap<Vec<MCTSMove>, f64> {
    tree.paths(parent)
        .into_iter()
        .map(|path| {
            let mut moves = path.iter().map(|mh| tree.mv(*mh).mov.clone()).collect_vec();
            moves.push(mv.clone());
            // Collect cost of choices so far.
            let mut score = path
                .iter()
                .map(|mh| -(tree.siblings(*mh).len() as f64).ln())
                .sum();
            println!("score {}", score);
            // Add cost of final choice.
            score -= (tree.node(parent).outgoing().len() as f64).ln();
            println!("score {}", score);
            // Leave reward at each node.
            score -= ((moves.len() + (children as usize)) as f64) * 2f64.ln();
            println!("score {}", score);
            (moves, score)
        })
        .collect()
}

pub struct BestSoFarMoveEvaluator;
pub struct RescaledByBestMoveEvaluator;
pub struct RelativeMassMoveEvaluator;
pub struct ThompsonMoveEvaluator;

pub struct MCTSStateEvaluator;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum MCTSMoveState {
    SampleRule(RuleContext),
    RegenerateRule(Option<(usize, RuleContext)>),
    Compose(Vec<Composition>),
    Recurse(Vec<Recursion>),
    Variablize(Vec<Variablization>),
    MemorizeDatum,
    DeleteRule,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MCTSMove {
    SampleRule,
    SampleAtom(Atom),
    RegenerateRule,
    RegenerateThisRule(usize, RuleContext),
    MemorizeDatum(Option<usize>),
    DeleteRule(Option<usize>),
    Variablize(Option<Variablization>),
    Compose(Option<Composition>),
    Recurse(Option<Recursion>),
    Generalize,
    AntiUnify,
    Stop,
}

#[derive(Debug, Clone)]
pub enum PlayoutState<T: std::fmt::Debug + Copy> {
    Untried(HashMap<Vec<MCTSMove>, f64>),
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

impl MCTSMove {
    fn pretty(&self, lex: &Lexicon) -> String {
        match *self {
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
impl std::fmt::Display for MCTSMove {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
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

impl Revision {
    pub fn new(
        trs: HypothesisHandle,
        spec: Option<MCTSMoveState>,
        n: usize,
        paths: HashMap<Vec<MCTSMove>, f64>,
    ) -> Self {
        Revision {
            trs,
            spec,
            n,
            playout: PlayoutState::Untried(paths),
        }
    }
    pub fn show(&self) {
        println!("n: {}", self.n);
        println!("trs: {}", self.trs);
        println!("playout: {:?}", self.playout);
        println!("spec: {:?}", self.spec);
    }
    fn available_moves_inner<'a, 'b>(
        trs: &TRS<'a, 'b>,
        spec: &Option<MCTSMoveState>,
        n: usize,
        mcts: &TRSMCTS,
    ) -> Vec<MCTSMove> {
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
                        println!("pushing MemorizeDatum");
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
                    trs.lex
                        .rulecontext_fillers(&context, &place)
                        .into_iter()
                        .map(MCTSMove::SampleAtom)
                        .for_each(|mv| moves.push(mv))
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
    pub fn available_moves(&self, mcts: &TRSMCTS) -> Vec<MCTSMove> {
        Revision::available_moves_inner(&mcts.trss[self.trs], &self.spec, self.n, mcts)
    }
    pub fn make_move_inner<'a, 'b>(
        mv: &MCTSMove,
        spec: &Option<MCTSMoveState>,
        data: &[Datum],
        trs: &TRS<'a, 'b>,
    ) -> MoveResult<'a, 'b> {
        match *mv {
            MCTSMove::Stop => MoveResult::Terminal,
            MCTSMove::Generalize => {
                let trs = tryo![trs.generalize().ok()];
                println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
                MoveResult::Revision(Some(trs), None)
            }
            MCTSMove::AntiUnify => {
                let trs = tryo![trs.lgg().ok()];
                println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
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
                println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
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
                println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
                MoveResult::Revision(Some(trs), None)
            }
            MCTSMove::Variablize(None) => {
                let types = trs.collect_types();
                let vs = trs.find_all_variablizations(&types);
                if vs.is_empty() {
                    MoveResult::Failed
                } else {
                    MoveResult::Revision(None, Some(MCTSMoveState::Variablize(vs)))
                }
            }
            MCTSMove::Variablize(Some((ref m, ref tp, ref places))) => {
                let mut clauses = trs.utrs.clauses();
                clauses[*m] = tryo![trs.apply_variablization_typeless(tp, places, &clauses[*m])];
                let new_trs = tryo![trs.adopt_solution(&mut clauses.clone())];
                println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
                MoveResult::Revision(Some(new_trs), None)
            }
            MCTSMove::DeleteRule(None) => {
                MoveResult::Revision(None, Some(MCTSMoveState::DeleteRule))
            }
            MCTSMove::DeleteRule(Some(n)) => {
                let mut trs = trs.clone();
                tryo![trs.utrs.remove_idx(n).ok()];
                println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
                MoveResult::Revision(Some(trs), None)
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
                println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
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
                    let new_context = tryo![context.replace(&place, Context::from(atom))];
                    if let Ok(rule) = Rule::try_from(&new_context) {
                        let mut trs = trs.clone();
                        tryo![trs.append_clauses(vec![rule]).ok()];
                        println!("#   \"{}\"", trs.to_string().lines().join(" "));
                        MoveResult::Revision(Some(trs), None)
                    } else {
                        let spec = Some(MCTSMoveState::SampleRule(new_context));
                        MoveResult::Revision(None, spec)
                    }
                }
                Some(MCTSMoveState::RegenerateRule(Some((n, ref context)))) => {
                    let place = tryo![context.leftmost_hole()];
                    let new_context = tryo![context.replace(&place, Context::from(atom))];
                    if let Ok(rule) = Rule::try_from(&new_context) {
                        let mut trs = trs.clone();
                        tryo![trs.utrs.remove_idx(*n).ok()];
                        tryo![trs.utrs.insert_idx(*n, rule).ok()];
                        println!("#   \"{}\"", trs.to_string().lines().join(" "));
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
    pub fn make_move(
        parent: NodeHandle,
        mv: &MCTSMove,
        mcts: &mut TRSMCTS,
        tree: &TreeStore<TRSMCTS>,
        handle: RevisionHandle,
    ) -> Option<(MCTSState, Option<usize>)> {
        let trsh = mcts.revisions[handle].trs;
        println!("#   move is {}", mv);
        println!(
            "#   trs is \"{}\"",
            mcts.trss[trsh].to_string().lines().join(" ")
        );
        let n = mcts.revisions[handle].n;
        let state = match Revision::make_move_inner(
            mv,
            &mcts.revisions[handle].spec,
            &mcts.data,
            &mcts.trss[trsh],
        ) {
            MoveResult::Failed => None,
            MoveResult::Terminal => {
                let paths = compute_paths(parent, mv, tree, false);
                let hh = mcts.find_hypothesis(mcts.trss[trsh].clone(), paths);
                Some(StateKind::Terminal(Terminal::new(hh)))
            }
            MoveResult::Revision(None, new_spec) => {
                let new_n = n + (new_spec.is_none() as usize);
                let children =
                    !Revision::available_moves_inner(&mcts.trss[trsh], &new_spec, new_n, mcts)
                        .is_empty();
                let paths = compute_paths(parent, mv, tree, children);
                Some(StateKind::Revision(Revision::new(
                    trsh, new_spec, new_n, paths,
                )))
            }
            MoveResult::Revision(Some(trs), new_spec) => {
                let new_n = n + (new_spec.is_none() as usize);
                let children =
                    !Revision::available_moves_inner(&trs, &new_spec, new_n, mcts).is_empty();
                let paths = compute_paths(parent, mv, tree, children);
                let hh = mcts.find_trs(trs);
                Some(StateKind::Revision(Revision::new(
                    hh, new_spec, new_n, paths,
                )))
            }
        }?;
        Some(mcts.add_state(state))
    }
    pub fn playout<'a, 'b, R: Rng>(
        rh: RevisionHandle,
        mcts: &mut TRSMCTS<'a, 'b>,
        paths: HashMap<Vec<MCTSMove>, f64>,
        rng: &mut R,
    ) -> Option<(TRS<'a, 'b>, HashMap<Vec<MCTSMove>, f64>)> {
        let trsh = mcts.revisions[rh].trs;
        let min_depth = paths.keys().map(|k| k.len()).min().unwrap();
        // TODO: Maybe try backtracking instead of a fixed count?
        for _ in 0..10 {
            let mut trs = mcts.trss[trsh].clone();
            let mut spec = mcts.revisions[rh].spec.clone();
            let mut n = mcts.revisions[rh].n;
            let mut progress = true;
            let mut meta_pairs = paths.iter().map(|(k, v)| (k.to_vec(), *v)).collect_vec();
            let mut depth = min_depth;
            while progress && depth < mcts.params.max_depth {
                progress = false;
                // Compute the available moves.
                let mut moves = Revision::available_moves_inner(&trs, &spec, n, mcts);
                // Choose a move (random policy).
                let moves_len = moves.len();
                println!("moves.len() = {}", moves_len);
                moves.shuffle(rng);
                for mv in moves {
                    println!("#         {}", mv);
                    match Revision::make_move_inner(&mv, &spec, &mcts.data, &trs) {
                        MoveResult::Failed => {
                            println!("move failed");
                            continue;
                        }
                        MoveResult::Terminal => {
                            println!("#           success");
                            let metas = meta_pairs
                                .into_iter()
                                .map(|(mut path, mut score)| {
                                    path.push(mv.clone());
                                    score -= (moves_len as f64).ln();
                                    println!("terminal: {:?} {}", path, score);
                                    (path, score)
                                })
                                .collect();
                            return Some((trs, metas));
                        }
                        MoveResult::Revision(new_trs, new_spec) => {
                            println!("#           success");
                            for mut pair in meta_pairs.iter_mut() {
                                pair.0.push(mv.clone());
                                pair.1 -= (moves_len as f64).ln() + 2f64.ln();
                                println!("revision: {:?} {}", pair.0, pair.1);
                            }
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
impl Eq for Revision {}
impl PartialEq for Revision {
    fn eq(&self, other: &Self) -> bool {
        self.trs == other.trs && self.spec == other.spec
    }
}

impl<'a, 'b> State<TRSMCTS<'a, 'b>> for MCTSState {
    type Move = MCTSMove;
    type MoveList = Vec<Self::Move>;
    type AbstractDepthAdjustment = usize;
    fn available_moves(&self, mcts: &mut TRSMCTS) -> Self::MoveList {
        match self.handle {
            StateHandle::Terminal(..) => vec![],
            StateHandle::Revision(rh) => mcts.revisions[rh].available_moves(mcts),
        }
    }
    fn make_move(
        &self,
        parent: NodeHandle,
        mv: &Self::Move,
        mcts: &mut TRSMCTS<'a, 'b>,
        tree: &TreeStore<TRSMCTS>,
    ) -> Option<(Self, Option<Self::AbstractDepthAdjustment>)> {
        match self.handle {
            StateHandle::Terminal(..) => panic!("cannot move from terminal"),
            StateHandle::Revision(rh) => Revision::make_move(parent, mv, mcts, tree, rh),
        }
    }
    fn add_moves_for_new_data(&self, moves: &[Self::Move], mcts: &mut TRSMCTS) -> Vec<Self::Move> {
        self.available_moves(mcts)
            .into_iter()
            .filter(|m| !moves.contains(&m))
            .collect()
    }
    fn adjust_abstract_depth(
        &self,
        adjustment: &Self::AbstractDepthAdjustment,
        mcts: &mut TRSMCTS,
    ) {
        match self.handle {
            StateHandle::Terminal(..) => (),
            StateHandle::Revision(rh) => mcts.revisions[rh].n -= *adjustment,
        }
    }
    fn describe_move(&self, mv: &Self::Move, mcts: &TRSMCTS) -> Value {
        match self.handle {
            StateHandle::Terminal(_) => Value::Null,
            StateHandle::Revision(rh) => {
                Value::String(mv.pretty(&mcts.trss[mcts.revisions[rh].trs].lex))
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
                    PlayoutState::Untried(_) => "untried".to_string(),
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

impl<'a, 'b> MCTS for TRSMCTS<'a, 'b> {
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

impl<'a, 'b> TRSMCTS<'a, 'b> {
    pub fn new(
        lexicon: Lexicon<'b>,
        bg: &'a [Rule],
        deterministic: bool,
        data: &'a [Datum],
        model: ModelParams,
        params: MCTSParams,
    ) -> TRSMCTS<'a, 'b> {
        TRSMCTS {
            lexicon,
            bg,
            deterministic,
            data,
            model,
            params,
            trss: vec![],
            hypotheses: vec![],
            terminals: vec![],
            revisions: vec![],
            best: std::f64::NEG_INFINITY,
        }
    }
    pub fn add_state(&mut self, state: StateKind) -> (MCTSState, Option<usize>) {
        match state {
            StateKind::Terminal(h) => (self.add_terminal(h), None),
            StateKind::Revision(r) => self.add_revision(r),
        }
    }
    pub fn add_revision(&mut self, state: Revision) -> (MCTSState, Option<usize>) {
        // Only create a new revision if you can't find this one already.
        let state_depth = state.n;
        let rh = match self.revisions.iter().position(|r| *r == state) {
            Some(rh) => rh,
            None => {
                self.revisions.push(state);
                self.revisions.len() - 1
            }
        };
        let adjustment = if state_depth < self.revisions[rh].n {
            Some(self.revisions[rh].n - state_depth)
        } else {
            None
        };
        let handle = StateHandle::Revision(rh);
        (MCTSState { handle }, adjustment)
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
    pub fn find_trs(&mut self, mut trs: TRS<'a, 'b>) -> TRSHandle {
        trs.utrs.canonicalize(&mut HashMap::new());
        match self.trss.iter().position(|t| TRS::same_shape(&t, &trs)) {
            Some(trsh) => trsh,
            None => {
                self.trss.push(trs);
                self.trss.len() - 1
            }
        }
    }
    pub fn find_hypothesis(
        &mut self,
        mut trs: TRS<'a, 'b>,
        paths: HashMap<Vec<MCTSMove>, f64>,
    ) -> HypothesisHandle {
        trs.utrs.canonicalize(&mut HashMap::new());
        // Find the hypothesis.
        let hh = match self
            .hypotheses
            .iter()
            .position(|h| TRS::same_shape(&h.object.trs, &trs))
        {
            Some(hh) => hh,
            None => {
                let object = MCTSObj::new(trs, HashMap::new());
                let model = MCTSModel::new(self.model);
                self.hypotheses.push(Hypothesis::new(object, model));
                self.hypotheses.len() - 1
            }
        };
        // Update the path information.
        for (path, prior) in paths.into_iter() {
            println!("should be inserting {:?} {}", path, prior);
            self.hypotheses[hh]
                .object
                .metas
                .entry(path)
                .or_insert(prior);
        }
        hh
    }
    pub fn root(&mut self) -> MCTSState {
        let mut trs = TRS::new_unchecked(&self.lexicon, self.deterministic, self.bg, vec![]);
        trs.identify_symbols();
        let mut metas = HashMap::new();
        metas.insert(vec![], 0.5f64.ln());
        let state = Revision {
            trs: self.find_trs(trs),
            spec: None,
            n: 0,
            playout: PlayoutState::Untried(metas),
        };
        self.add_revision(state).0
    }
    pub fn update_ps(&mut self) {
        let mut hypotheses = std::mem::replace(&mut self.hypotheses, vec![]);
        for hypothesis in hypotheses.iter_mut() {
            for (path, prior) in hypothesis.object.metas.iter_mut() {
                *prior = self.p_path(path);
            }
            hypothesis.log_posterior(self.data);
            println!(
                "#       {:.4}\t\"{}\"",
                hypothesis.lposterior,
                hypothesis.object.trs.to_string().lines().join(" "),
            );
        }
        self.hypotheses = hypotheses;
    }
    pub fn p_path(&self, moves: &[MCTSMove]) -> f64 {
        // Set the root state.
        let mut trs = self.trss[self.revisions[0].trs].clone();
        let mut spec = None;
        let mut n = 0;
        let mut ps = Vec::with_capacity(moves.len());
        println!("moves: {:?}", moves);
        for mv in moves {
            // Ensure our move is available.
            let available_moves = Revision::available_moves_inner(&trs, &spec, n, self);
            println!("available_moves: {:?}", available_moves);
            if !available_moves.contains(mv) {
                println!("can't find move");
                return std::f64::NEG_INFINITY;
            }
            // Take the move and update the state accordingly.
            match Revision::make_move_inner(&mv, &spec, &self.data, &trs) {
                MoveResult::Failed => {
                    println!("move failed");
                    return std::f64::NEG_INFINITY;
                }
                MoveResult::Terminal => {
                    println!("terminating");
                    ps.push(available_moves.len() as f64);
                    break;
                }
                MoveResult::Revision(None, new_spec) => {
                    println!("no change to TRS");
                    ps.push(available_moves.len() as f64);
                    n += new_spec.is_none() as usize;
                    spec = new_spec;
                }
                MoveResult::Revision(Some(new_trs), new_spec) => {
                    println!("new TRS");
                    ps.push(available_moves.len() as f64);
                    n += new_spec.is_none() as usize;
                    spec = new_spec;
                    trs = new_trs;
                }
            }
        }
        println!("ps: {:?}", ps);
        if ps.len() == moves.len() {
            let prior = ps.into_iter().map(|p| -(2.0 * p).ln()).sum();
            println!("new prior: {}", prior);
            prior
        } else {
            println!("missing ps");
            std::f64::NEG_INFINITY
        }
    }
}

fn unexplored_first<'c, F, M: MCTS, R: Rng, MoveIter>(
    moves: MoveIter,
    nh: NodeHandle,
    tree: &SearchTree<M>,
    selector: F,
    rng: &mut R,
) -> Option<&'c MoveInfo<M>>
where
    MoveIter: Iterator<Item = &'c MoveInfo<M>>,
    F: Fn(&Stats<M>, &Stats<M>, &M, &mut R) -> f64,
{
    // Split the moves into those with and without children.
    let (childful, childless): (Vec<_>, Vec<_>) = moves.partition(|mv| mv.child.is_some());
    // Take the first childless move, or perform UCT on childed moves.
    if let Some(mv) = childless.choose(rng) {
        println!(
            "#   There are {} childless. We chose: {}.",
            childless.len(),
            mv.mov
        );
        Some(mv)
    } else {
        childful
            .into_iter()
            .map(|mv| {
                let ch = mv.child.expect("INVARIANT: partition failed us");
                let parent = &tree.node(nh).stats;
                let child = &tree.node(ch).stats;
                let score = selector(parent, child, tree.mcts(), rng);
                println!("#     ({}): {:.4} - {}", ch, score, mv.mov,);
                (mv, score)
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
            .map(|(mv, _)| {
                println!("#     we're going with {}", mv.mov);
                *mv
            })
            .or_else(|| {
                println!("#     no available moves");
                None
            })
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
    fn choose<'c, R: Rng, MoveIter>(
        &self,
        moves: MoveIter,
        nh: NodeHandle,
        tree: &SearchTree<TRSMCTS<'a, 'b>>,
        rng: &mut R,
    ) -> Option<&'c MoveInfo<TRSMCTS<'a, 'b>>>
    where
        MoveIter: Iterator<Item = &'c MoveInfo<TRSMCTS<'a, 'b>>>,
    {
        unexplored_first(moves, nh, tree, thompson_sample, rng)
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
                PlayoutState::Untried(_) => panic!("shouldn't reread empty state"),
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
        println!("#     evaluating");
        let score = match state.handle {
            StateHandle::Terminal(th) => {
                println!(
                    "#       node is terminal: {}",
                    mcts.hypotheses[mcts.terminals[th].trs]
                        .object
                        .trs
                        .to_string()
                        .lines()
                        .join(" ")
                );
                mcts.hypotheses[mcts.terminals[th].trs].log_posterior(mcts.data);
                println!("lprior: {}", mcts.hypotheses[mcts.terminals[th].trs].lprior);
                println!(
                    "lposterior: {}",
                    mcts.hypotheses[mcts.terminals[th].trs].lposterior
                );
                mcts.hypotheses[mcts.terminals[th].trs].lposterior
            }
            StateHandle::Revision(rh) => match mcts.revisions[rh].playout.clone() {
                PlayoutState::Untried(paths) => {
                    println!(
                        "#       playing out {}",
                        mcts.trss[mcts.revisions[rh].trs]
                            .to_string()
                            .lines()
                            .join(" ")
                    );
                    if let Some((trs, metas)) = Revision::playout(rh, mcts, paths, rng) {
                        println!(
                            "#         simulated: \"{}\"",
                            trs.to_string().lines().join(" ")
                        );
                        println!("feeding in metas: {:?}", metas);
                        let hh = mcts.find_hypothesis(trs, metas);
                        mcts.revisions[rh].playout = PlayoutState::Success(hh);
                        mcts.hypotheses[hh].log_posterior(mcts.data)
                    } else {
                        println!("playout failed");
                        mcts.revisions[rh].playout = PlayoutState::Failed;
                        std::f64::NEG_INFINITY
                    }
                }
                _ => panic!("should only evaluate a state once"),
            },
        };
        if score > mcts.best {
            mcts.best = score;
        }
        score
    }
}

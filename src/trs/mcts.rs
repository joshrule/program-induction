use itertools::Itertools;
use mcts::{
    MoveEvaluator, MoveInfo, NodeHandle, NodeStatistic, SearchTree, State, StateEvaluator, Stats,
    MCTS,
};
use polytype::TypeSchema;
use rand::{
    distributions::{Bernoulli, WeightedIndex},
    prelude::{Distribution, Rng, SliceRandom},
};
use serde_json::Value;
use std::{cmp::Ordering, collections::HashMap, convert::TryFrom};
use term_rewriting::{Atom, Context, Rule, RuleContext, Term};
use trs::{
    Composition, GenerationLimit, Hypothesis, Lexicon, ModelParams, Recursion, Types,
    Variablization, TRS,
};
use utils::{exp_normalize, logsumexp};

type RevisionHandle = usize;
type TerminalHandle = usize;
type HypothesisHandle = usize;

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
    trs: HypothesisHandle,
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

pub struct TRSMCTS<'a, 'b> {
    pub lexicon: Lexicon<'b>,
    pub bg: &'a [Rule],
    pub deterministic: bool,
    pub data: &'a [Rule],
    pub input: Option<&'a Term>,
    pub hypotheses: Vec<Hypothesis<'a, 'b>>,
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
    let source = if Bernoulli::from_ratio(n, m + n).unwrap().sample(rng) {
        child
    } else {
        parent
    };
    exp_normalize(source, Some(-1000f64.ln()))
        .map(|ps| source[WeightedIndex::new(ps).unwrap().sample(rng)])
        .unwrap_or_else(|| *source.choose(rng).unwrap())
}

pub struct BestSoFarMoveEvaluator;
pub struct RescaledByBestMoveEvaluator;
pub struct RelativeMassMoveEvaluator;
pub struct ThompsonMoveEvaluator;

pub struct MCTSStateEvaluator;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum MCTSMoveState {
    Compose(Vec<Composition>),
    Recurse(Vec<Recursion>),
    Variablize(Vec<Variablization>, Types),
    VariablizeRule(RevisionHandle, Option<usize>, Vec<Rule>),
    MemorizeData(Option<usize>),
    SampleRule(RuleContext),
    RegenerateRule(Option<(usize, RuleContext)>),
    DeleteRules(Option<usize>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MCTSMove {
    MemorizeData,
    MemorizeDatum(Option<usize>),
    SampleRule,
    SampleAtom(Atom),
    RegenerateRule,
    RegenerateThisRule(usize, RuleContext),
    DeleteRules,
    DeleteRule(Option<usize>),
    Generalize,
    AntiUnify,
    Variablize,
    VariablizeRule(Option<usize>),
    Compose(Option<Composition>),
    Recurse(Option<Recursion>),
    Stop,
}

#[derive(Debug, Clone, Copy)]
pub enum PlayoutState<T: std::fmt::Debug + Copy> {
    Untried,
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
    fn pretty(&self, lex: &Lexicon, _data: &[Rule]) -> String {
        match *self {
            MCTSMove::MemorizeDatum(Some(n)) => format!("MemorizeDatum({})", n),
            MCTSMove::SampleAtom(atom) => format!("SampleAtom({})", atom.display(lex.signature())),
            MCTSMove::RegenerateThisRule(n, ref c) => {
                format!("RegenerateThisRule({}, {})", n, c.pretty(lex.signature()))
            }
            MCTSMove::DeleteRule(Some(n)) => format!("DeleteRule({})", n),
            MCTSMove::VariablizeRule(Some(n)) => format!("Variablize({})", n),
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
            MCTSMove::Variablize => "Variablize".to_string(),
            MCTSMove::VariablizeRule(None) => "VariablizeRule(Stop)".to_string(),
            MCTSMove::MemorizeDatum(None) => "MemorizeDatum(Stop)".to_string(),
            MCTSMove::DeleteRule(None) => "DeleteRule(Stop)".to_string(),
            MCTSMove::MemorizeData => "MemorizeData".to_string(),
            MCTSMove::SampleRule => "SampleRule".to_string(),
            MCTSMove::RegenerateRule => "RegenerateRule".to_string(),
            MCTSMove::DeleteRules => "DeleteRules".to_string(),
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
            MCTSMove::MemorizeDatum(n) => write!(f, "MemorizeDatum({:?})", n),
            MCTSMove::VariablizeRule(n) => write!(f, "VariablizeRule({:?})", n),
            MCTSMove::SampleAtom(atom) => write!(f, "SampleAtom({:?})", atom),
            MCTSMove::RegenerateThisRule(n, _) => write!(f, "RegenerateThisRule({}, context)", n),
            MCTSMove::DeleteRule(n) => write!(f, "DeleteRule({:?})", n),
            MCTSMove::MemorizeData => write!(f, "MemorizeData"),
            MCTSMove::SampleRule => write!(f, "SampleRule"),
            MCTSMove::RegenerateRule => write!(f, "RegenerateRule"),
            MCTSMove::DeleteRules => write!(f, "DeleteRules"),
            MCTSMove::Generalize => write!(f, "Generalize"),
            MCTSMove::AntiUnify => write!(f, "AntiUnify"),
            MCTSMove::Variablize => write!(f, "Variablize"),
            MCTSMove::Compose(None) => write!(f, "Compose"),
            MCTSMove::Recurse(None) => write!(f, "Recurse"),
            MCTSMove::Compose(_) => write!(f, "Compose(_)"),
            MCTSMove::Recurse(_) => write!(f, "Recurse(_)"),
            MCTSMove::Stop => write!(f, "Stop"),
        }
    }
}

struct MoveDist(WeightedIndex<u8>);

pub fn take_mcts_step<'a, 'b, R: Rng>(
    mut maybe_trs: Option<TRS<'a, 'b>>,
    steps_remaining: &mut usize,
    mcts: &TRSMCTS<'a, 'b>,
    rng: &mut R,
) -> Option<TRS<'a, 'b>> {
    let mut trs = maybe_trs.take()?;
    trs.utrs.canonicalize(&mut HashMap::new());
    let move_dist = MoveDist::new(&trs, &mcts.data);
    match move_dist.sample(rng) {
        MCTSMove::MemorizeData => {
            println!("#         adding data");
            let mut order_first = (0..mcts.data.len()).collect_vec();
            order_first.shuffle(rng);
            let mut lower_bound = None;
            for first in order_first {
                if trs.append_clauses(vec![mcts.data[first].clone()]).is_ok() {
                    lower_bound.replace(first + 1);
                    break;
                }
            }
            // We don't mind failure now that we have one success.
            for rule in mcts.data.iter().skip(lower_bound?) {
                if rng.gen() {
                    trs.append_clauses(vec![rule.clone()]).ok();
                }
            }
            println!("#           success");
            *steps_remaining = steps_remaining.saturating_sub(1);
            return Some(trs);
        }
        MCTSMove::SampleRule => {
            let schema = TypeSchema::Monotype(trs.lex.0.to_mut().ctx.new_variable());
            println!("#         sampling a rule");
            for _ in 0..100 {
                println!("#           looping");
                let mut ctx = trs.lex.0.ctx.clone();
                if let Ok(rule) = trs.lex.sample_rule(
                    &schema,
                    mcts.params.atom_weights,
                    GenerationLimit::TermSize(mcts.params.max_size),
                    mcts.params.invent,
                    &mut ctx,
                    rng,
                ) {
                    if trs.append_clauses(vec![rule.clone()]).is_ok() {
                        println!("#           success: {}", rule.pretty(&trs.lex.signature()));
                        *steps_remaining = steps_remaining.saturating_sub(1);
                        return Some(trs);
                    }
                }
            }
            None
        }
        MCTSMove::RegenerateRule => {
            let mut idxs = (0..trs.len()).collect_vec();
            idxs.shuffle(rng);
            for idx in idxs {
                let rulecontext = RuleContext::from(trs.utrs.rules[idx].clone());
                let mut subcontexts = rulecontext
                    .subcontexts()
                    .into_iter()
                    .map(|(_, place)| place)
                    .collect_vec();
                subcontexts.shuffle(rng);
                for place in subcontexts {
                    if let Some(mut context) = rulecontext.replace(&place, Context::Hole) {
                        if RuleContext::is_valid(&context.lhs, &context.rhs) {
                            context.canonicalize(&mut HashMap::new());
                            println!(
                                "#         regenerating: {}",
                                context.pretty(trs.lex.signature())
                            );
                            for _ in 0..100 {
                                println!("#           looping");
                                if let Ok(rule) = trs.lex.sample_rule_from_context(
                                    &context,
                                    mcts.params.atom_weights,
                                    GenerationLimit::TotalSize(
                                        context.size() + mcts.params.max_size - 1,
                                    ),
                                    mcts.params.invent,
                                    &mut trs.lex.0.ctx.clone(),
                                    rng,
                                ) {
                                    let old_rule = trs.utrs.rules[idx].clone();
                                    let mut new_trs = trs.clone();
                                    if new_trs.utrs.replace(idx, &old_rule, rule.clone()).is_ok() {
                                        println!(
                                            "#           success: {}",
                                            rule.pretty(&new_trs.lex.signature())
                                        );
                                        *steps_remaining = steps_remaining.saturating_sub(1);
                                        return Some(new_trs);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            None
        }
        MCTSMove::DeleteRules => {
            println!("#         deleting rules");
            let mut upper_bound = None;
            let mut order_first = (0..trs.len()).collect_vec();
            order_first.shuffle(rng);
            for first in order_first {
                if trs.utrs.remove_idx(first).is_ok() {
                    upper_bound.replace(first);
                    break;
                }
            }
            // Failure is okay given one success.
            for idx in (0..upper_bound?).rev() {
                if rng.gen() {
                    trs.utrs.remove_idx(idx).ok();
                }
            }
            println!("#           success");
            *steps_remaining = steps_remaining.saturating_sub(1);
            Some(trs)
        }
        MCTSMove::Variablize => {
            println!("#         variablizing");
            let mut types = trs.collect_types();
            let vs = trs.find_all_variablizations(&types);
            let mut clauses = trs.utrs.clauses();
            let mut order_first = (0..vs.len()).collect_vec();
            order_first.shuffle(rng);
            let mut maybe_first = None;
            for first in order_first {
                let (m, tp, places) = &vs[first];
                if let Some(rule) = trs.apply_variablization(&tp, &places, &clauses[*m], &mut types)
                {
                    clauses[*m] = rule;
                    if let Some(new_trs) = trs.adopt_solution(&mut clauses.clone()) {
                        trs = new_trs;
                        maybe_first.replace(first);
                        break;
                    }
                }
            }
            // Failure is okay given one success.
            for (m, tp, places) in vs.iter().skip(maybe_first? + 1) {
                if rng.gen() {
                    if let Some(rule) =
                        trs.apply_variablization(&tp, &places, &clauses[*m], &mut types)
                    {
                        clauses[*m] = rule;
                        if let Some(new_trs) = trs.adopt_solution(&mut clauses.clone()) {
                            trs = new_trs;
                        }
                    }
                }
            }
            println!("#           success");
            *steps_remaining = steps_remaining.saturating_sub(1);
            Some(trs)
        }
        MCTSMove::Generalize => {
            println!("#         generalizing");
            let new_trs = trs.generalize().ok()?;
            println!("#           success");
            *steps_remaining = steps_remaining.saturating_sub(1);
            Some(new_trs)
        }
        MCTSMove::Compose(None) => {
            println!("#         composing");
            let mut compositions = trs.find_all_compositions();
            compositions.shuffle(rng);
            for composition in compositions {
                if let Some(new_trs) = trs.compose_by(&composition) {
                    println!("#        success");
                    *steps_remaining = steps_remaining.saturating_sub(1);
                    return Some(new_trs);
                }
            }
            None
        }
        MCTSMove::Recurse(None) => {
            println!("#         recursing");
            let mut recursions = trs.find_all_recursions();
            recursions.shuffle(rng);
            for recursion in recursions {
                if let Some(new_trs) = trs.recurse_by(&recursion) {
                    println!("#        success");
                    *steps_remaining = steps_remaining.saturating_sub(1);
                    return Some(new_trs);
                }
            }
            None
        }
        MCTSMove::AntiUnify => {
            println!("#         anti-unifying");
            let new_trs = trs.lgg().ok()?;
            println!("#           success");
            *steps_remaining = steps_remaining.saturating_sub(1);
            Some(new_trs)
        }
        MCTSMove::Stop => {
            println!("#         stopping");
            println!("#           success");
            *steps_remaining = 0;
            Some(trs)
        }
        _ => None,
    }
}

impl MoveDist {
    fn new(trs: &TRS, data: &[Rule]) -> Self {
        let has_data_weight = !data.is_empty() as u8;
        let non_empty_trs_weight = !trs.is_empty() as u8;
        let two_plus_trs_weight = (trs.len() > 1) as u8;
        let dist = WeightedIndex::new(&[
            // Stop
            1,
            // Sample Rule
            1,
            // Regenerate
            non_empty_trs_weight,
            // Delete
            two_plus_trs_weight,
            // Memorize Data
            has_data_weight,
            // Generalize
            non_empty_trs_weight,
            // Compose
            non_empty_trs_weight,
            // Recurse
            non_empty_trs_weight,
            // Variablize
            non_empty_trs_weight,
            // AntiUnify
            two_plus_trs_weight,
        ])
        .unwrap();
        MoveDist(dist)
    }
}

impl Distribution<MCTSMove> for MoveDist {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MCTSMove {
        match self.0.sample(rng) {
            0 => MCTSMove::Stop,
            1 => MCTSMove::SampleRule,
            2 => MCTSMove::RegenerateRule,
            3 => MCTSMove::DeleteRules,
            4 => MCTSMove::MemorizeData,
            5 => MCTSMove::Generalize,
            6 => MCTSMove::Compose(None),
            7 => MCTSMove::Recurse(None),
            8 => MCTSMove::Variablize,
            9 => MCTSMove::AntiUnify,
            _ => unreachable!(),
        }
    }
}

impl Terminal {
    pub fn new(trs: HypothesisHandle) -> Self {
        Terminal { trs }
    }
}

impl Revision {
    pub fn new(trs: HypothesisHandle, spec: Option<MCTSMoveState>, n: usize) -> Self {
        Revision {
            trs,
            spec,
            n,
            playout: PlayoutState::Untried,
        }
    }
    pub fn show(&self) {
        println!("n: {}", self.n);
        println!("trs: {}", self.trs);
        println!("playout: {:?}", self.playout);
        println!("spec: {:?}", self.spec);
    }
    pub fn available_moves(&self, mcts: &TRSMCTS, moves: &mut Vec<MCTSMove>, _rh: RevisionHandle) {
        let trs = &mcts.hypotheses[self.trs].trs;
        match self.spec {
            None => {
                // Search can always stop.
                moves.push(MCTSMove::Stop);
                if self.n < mcts.params.max_revisions {
                    // Search can always sample a new rule.
                    moves.push(MCTSMove::SampleRule);
                    // A TRS must have a rule in order to regenerate or generalize.
                    if !trs.is_empty() {
                        moves.push(MCTSMove::RegenerateRule);
                        moves.push(MCTSMove::Generalize);
                        moves.push(MCTSMove::Compose(None));
                        moves.push(MCTSMove::Recurse(None));
                        moves.push(MCTSMove::Variablize);
                    }
                    // A TRS must have >1 rule to delete without creating cycles.
                    // Anti-unification relies on having two rules to unify.
                    if trs.len() > 1 {
                        moves.push(MCTSMove::DeleteRules);
                        moves.push(MCTSMove::AntiUnify);
                    }
                    // We can only add data if there's data to add.
                    if mcts
                        .data
                        .iter()
                        .filter(|datum| trs.utrs.get_clause(datum).is_none())
                        .count()
                        > 0
                    {
                        moves.push(MCTSMove::MemorizeData);
                    }
                }
            }
            Some(MCTSMoveState::Variablize(ref vs, _)) => {
                (0..vs.len()).for_each(|n| moves.push(MCTSMove::VariablizeRule(Some(n))))
            }
            Some(MCTSMoveState::VariablizeRule(rh, n, _)) => {
                let lower_bound = n.unwrap_or(0);
                match mcts.revisions[rh].spec {
                    Some(MCTSMoveState::Variablize(ref vs, _)) => {
                        (lower_bound..vs.len())
                            .map(|i_var| MCTSMove::VariablizeRule(Some(i_var)))
                            .for_each(|mv| moves.push(mv));
                        if n.is_some() {
                            moves.push(MCTSMove::VariablizeRule(None));
                        }
                    }
                    _ => panic!("Variablize cannot find reference data"),
                }
            }
            Some(MCTSMoveState::Compose(ref compositions)) => compositions
                .iter()
                .cloned()
                .for_each(|composition| moves.push(MCTSMove::Compose(Some(composition)))),
            Some(MCTSMoveState::Recurse(ref recursions)) => recursions
                .iter()
                .cloned()
                .for_each(|recursion| moves.push(MCTSMove::Recurse(Some(recursion)))),
            Some(MCTSMoveState::MemorizeData(n)) => {
                let lower_bound = n.unwrap_or(0);
                (lower_bound..mcts.data.len())
                    .filter(|i_datum| trs.utrs.get_clause(&mcts.data[*i_datum]).is_none())
                    .map(|i_datum| MCTSMove::MemorizeDatum(Some(i_datum)))
                    .for_each(|mv| moves.push(mv));
                if n.is_some() {
                    moves.push(MCTSMove::MemorizeDatum(None));
                }
            }
            Some(MCTSMoveState::DeleteRules(n)) => {
                let upper_bound = n.unwrap_or_else(|| trs.len());
                (0..upper_bound)
                    .map(|rule| MCTSMove::DeleteRule(Some(rule)))
                    .for_each(|mv| moves.push(mv));
                if n.is_some() {
                    moves.push(MCTSMove::DeleteRule(None));
                }
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
    }
    pub fn make_move(
        &self,
        mv: &MCTSMove,
        mcts: &mut TRSMCTS,
        self_handle: RevisionHandle,
    ) -> Option<StateKind> {
        let trs = &mcts.hypotheses[self.trs].trs;
        println!("#   move is {}", mv);
        println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
        match *mv {
            MCTSMove::Stop => Some(StateKind::Terminal(Terminal::new(self.trs))),
            MCTSMove::Generalize => {
                let trs = trs.generalize().ok()?;
                println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
                let hh = mcts.find_hypothesis(trs);
                Some(StateKind::Revision(Revision::new(hh, None, self.n + 1)))
            }
            MCTSMove::AntiUnify => {
                let trs = trs.lgg().ok()?;
                println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
                let hh = mcts.find_hypothesis(trs);
                Some(StateKind::Revision(Revision::new(hh, None, self.n + 1)))
            }
            MCTSMove::Compose(None) => {
                let compositions = trs.find_all_compositions();
                if compositions.is_empty() {
                    None
                } else {
                    Some(StateKind::Revision(Revision::new(
                        self.trs,
                        Some(MCTSMoveState::Compose(compositions)),
                        self.n,
                    )))
                }
            }
            MCTSMove::Compose(Some(ref composition)) => {
                let trs = trs.compose_by(composition)?;
                println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
                let hh = mcts.find_hypothesis(trs);
                Some(StateKind::Revision(Revision::new(hh, None, self.n + 1)))
            }
            MCTSMove::Recurse(None) => {
                let recursions = trs.find_all_recursions();
                if recursions.is_empty() {
                    None
                } else {
                    Some(StateKind::Revision(Revision::new(
                        self.trs,
                        Some(MCTSMoveState::Recurse(recursions)),
                        self.n,
                    )))
                }
            }
            MCTSMove::Recurse(Some(ref recursion)) => {
                let trs = trs.recurse_by(recursion)?;
                println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
                let hh = mcts.find_hypothesis(trs);
                Some(StateKind::Revision(Revision::new(hh, None, self.n + 1)))
            }
            MCTSMove::Variablize => {
                let types = trs.collect_types();
                let vs = trs.find_all_variablizations(&types);
                if vs.is_empty() {
                    None
                } else {
                    Some(StateKind::Revision(Revision::new(
                        self.trs,
                        Some(MCTSMoveState::Variablize(vs, types)),
                        self.n,
                    )))
                }
            }
            MCTSMove::VariablizeRule(None) => Some(StateKind::Revision(Revision::new(
                self.trs,
                None,
                self.n + 1,
            ))),
            MCTSMove::VariablizeRule(Some(n)) => match self.spec {
                Some(MCTSMoveState::VariablizeRule(rh, _, ref rs)) => match mcts.revisions[rh].spec
                {
                    Some(MCTSMoveState::Variablize(ref vs, ref mut types)) => {
                        let mut clauses = rs.clone();
                        let (m, tp, places) = &vs[n];
                        clauses[*m] =
                            trs.apply_variablization(&tp, &places, &clauses[*m], types)?;
                        let new_trs = trs.adopt_solution(&mut clauses.clone())?;
                        println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
                        let hh = mcts.find_hypothesis(new_trs);
                        let spec = Some(MCTSMoveState::VariablizeRule(rh, Some(n + 1), clauses));
                        Some(StateKind::Revision(Revision::new(hh, spec, self.n + 1)))
                    }
                    _ => panic!("Variablize cannot find reference data"),
                },
                Some(MCTSMoveState::Variablize(ref vs, ref types)) => {
                    let trs = &mcts.hypotheses[self.trs].trs;
                    let mut types = types.clone();
                    let mut clauses = trs.utrs.clauses();
                    let (m, tp, places) = &vs[n];
                    clauses[*m] =
                        trs.apply_variablization(&tp, &places, &clauses[*m], &mut types)?;
                    let new_trs = trs.adopt_solution(&mut clauses.clone())?;
                    println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
                    let hh = mcts.find_hypothesis(new_trs);
                    let spec = Some(MCTSMoveState::VariablizeRule(
                        self_handle,
                        Some(n + 1),
                        clauses,
                    ));
                    Some(StateKind::Revision(Revision::new(hh, spec, self.n + 1)))
                }
                _ => panic!("Variablize Move and MoveState mismatch: {:?}", self.spec),
            },
            MCTSMove::DeleteRules => Some(StateKind::Revision(Revision::new(
                self.trs,
                Some(MCTSMoveState::DeleteRules(None)),
                self.n,
            ))),
            MCTSMove::DeleteRule(None) => Some(StateKind::Revision(Revision::new(
                self.trs,
                None,
                self.n + 1,
            ))),
            MCTSMove::DeleteRule(Some(n)) => {
                let mut trs = trs.clone();
                trs.utrs.remove_idx(n).ok()?;
                println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
                let hh = mcts.find_hypothesis(trs);
                let spec = Some(MCTSMoveState::DeleteRules(Some(n)));
                Some(StateKind::Revision(Revision::new(hh, spec, self.n)))
            }
            MCTSMove::MemorizeData => Some(StateKind::Revision(Revision::new(
                self.trs,
                Some(MCTSMoveState::MemorizeData(None)),
                self.n,
            ))),
            MCTSMove::MemorizeDatum(None) => Some(StateKind::Revision(Revision::new(
                self.trs,
                None,
                self.n + 1,
            ))),
            MCTSMove::MemorizeDatum(Some(n)) => {
                let mut trs = trs.clone();
                trs.append_clauses(vec![mcts.data[n].clone()]).ok()?;
                println!("#   trs is \"{}\"", trs.to_string().lines().join(" "));
                let hh = mcts.find_hypothesis(trs);
                let spec = Some(MCTSMoveState::MemorizeData(Some(n + 1)));
                Some(StateKind::Revision(Revision::new(hh, spec, self.n)))
            }
            MCTSMove::SampleRule => {
                let spec = Some(MCTSMoveState::SampleRule(RuleContext::default()));
                Some(StateKind::Revision(Revision::new(self.trs, spec, self.n)))
            }
            MCTSMove::RegenerateRule => {
                let spec = Some(MCTSMoveState::RegenerateRule(None));
                Some(StateKind::Revision(Revision::new(self.trs, spec, self.n)))
            }
            MCTSMove::RegenerateThisRule(n, ref context) => {
                let spec = Some(MCTSMoveState::RegenerateRule(Some((n, context.clone()))));
                Some(StateKind::Revision(Revision::new(self.trs, spec, self.n)))
            }
            MCTSMove::SampleAtom(atom) => match self.spec {
                Some(MCTSMoveState::SampleRule(ref context)) => {
                    let place = context.leftmost_hole()?;
                    let new_context = context.replace(&place, Context::from(atom))?;
                    if let Ok(rule) = Rule::try_from(&new_context) {
                        let mut trs = trs.clone();
                        trs.append_clauses(vec![rule]).ok()?;
                        println!("#   \"{}\"", trs.to_string().lines().join(" "));
                        let hh = mcts.find_hypothesis(trs);
                        Some(StateKind::Revision(Revision::new(hh, None, self.n)))
                    } else {
                        let spec = Some(MCTSMoveState::SampleRule(new_context));
                        Some(StateKind::Revision(Revision::new(self.trs, spec, self.n)))
                    }
                }
                Some(MCTSMoveState::RegenerateRule(Some((n, ref context)))) => {
                    let place = context.leftmost_hole()?;
                    let new_context = context.replace(&place, Context::from(atom))?;
                    if let Ok(rule) = Rule::try_from(&new_context) {
                        let mut trs = trs.clone();
                        trs.utrs.remove_idx(n).ok()?;
                        trs.utrs.insert_idx(n, rule).ok()?;
                        println!("#   \"{}\"", trs.to_string().lines().join(" "));
                        let hh = mcts.find_hypothesis(trs);
                        Some(StateKind::Revision(Revision::new(hh, None, self.n)))
                    } else {
                        let spec = Some(MCTSMoveState::RegenerateRule(Some((n, new_context))));
                        Some(StateKind::Revision(Revision::new(self.trs, spec, self.n)))
                    }
                }
                _ => panic!("MCTSMoveState doesn't match MCTSMove"),
            },
        }
    }
    pub fn playout<'a, 'b, R: Rng>(
        &self,
        mcts: &TRSMCTS<'a, 'b>,
        rng: &mut R,
    ) -> Option<TRS<'a, 'b>> {
        let mut steps_remaining = mcts.params.max_revisions.saturating_sub(self.n);
        let mut trs = mcts.hypotheses[self.trs].trs.clone();
        trs = self.finish_step_in_progress(Some(trs), &mut steps_remaining, mcts, rng)?;
        let mut count = 0;
        while steps_remaining > 0 && count < mcts.params.max_revisions * 3 {
            if let Some(new_trs) =
                take_mcts_step(Some(trs.clone()), &mut steps_remaining, mcts, rng)
            {
                trs = new_trs;
            }
            count += 1;
        }
        // It may not have actually hit "STOP", but it at least completed the move.
        Some(trs)
    }
    fn finish_step_in_progress<'a, 'b, R: Rng>(
        &self,
        mut maybe_trs: Option<TRS<'a, 'b>>,
        steps_remaining: &mut usize,
        mcts: &TRSMCTS,
        rng: &mut R,
    ) -> Option<TRS<'a, 'b>> {
        let mut trs = maybe_trs.take()?;
        trs.utrs.canonicalize(&mut HashMap::new());
        match &self.spec {
            None => return Some(trs),
            Some(MCTSMoveState::Compose(ref compositions)) => {
                println!("#        finishing composition");
                let mut compositions = compositions.clone();
                compositions.shuffle(rng);
                for composition in compositions {
                    if let Some(new_trs) = trs.compose_by(&composition) {
                        println!("#        success");
                        *steps_remaining = steps_remaining.saturating_sub(1);
                        return Some(new_trs);
                    }
                }
            }
            Some(MCTSMoveState::Recurse(ref recursions)) => {
                println!("#        finishing recursion");
                let mut recursions = recursions.clone();
                recursions.shuffle(rng);
                for recursion in recursions {
                    if let Some(new_trs) = trs.recurse_by(&recursion) {
                        println!("#        success");
                        *steps_remaining = steps_remaining.saturating_sub(1);
                        return Some(new_trs);
                    }
                }
            }
            Some(MCTSMoveState::DeleteRules(progress)) => {
                println!("#        finishing deletion");
                let mut upper_bound = *progress;
                if upper_bound.is_none() {
                    let mut order_first = (0..trs.len()).collect_vec();
                    order_first.shuffle(rng);
                    for first in order_first {
                        if trs.utrs.remove_idx(first).is_ok() {
                            upper_bound.replace(first);
                            break;
                        }
                    }
                }
                // Failure is okay given one success.
                for idx in (0..upper_bound?).rev() {
                    if rng.gen() {
                        trs.utrs.remove_idx(idx).ok();
                    }
                }
                println!("#        success");
                *steps_remaining = steps_remaining.saturating_sub(1);
                return Some(trs);
            }
            Some(MCTSMoveState::Variablize(ref vs, ref types)) => {
                println!("#        finishing variablization");
                let mut types = types.clone();
                let mut clauses = trs.utrs.clauses();
                let mut order_first = (0..vs.len()).collect_vec();
                order_first.shuffle(rng);
                let mut maybe_first = None;
                for first in order_first {
                    let (m, tp, places) = &vs[first];
                    if let Some(rule) =
                        trs.apply_variablization(&tp, &places, &clauses[*m], &mut types)
                    {
                        clauses[*m] = rule;
                        if let Some(new_trs) = trs.adopt_solution(&mut clauses.clone()) {
                            trs = new_trs;
                            maybe_first.replace(first);
                            break;
                        }
                    }
                }
                for (m, tp, places) in vs.iter().skip(maybe_first? + 1) {
                    if rng.gen() {
                        if let Some(rule) =
                            trs.apply_variablization(&tp, &places, &clauses[*m], &mut types)
                        {
                            clauses[*m] = rule;
                            if let Some(new_trs) = trs.adopt_solution(&mut clauses.clone()) {
                                trs = new_trs;
                            }
                        }
                    }
                }
                println!("#        success");
                *steps_remaining = steps_remaining.saturating_sub(1);
                return Some(trs);
            }
            Some(MCTSMoveState::VariablizeRule(rh, n, rs)) => {
                println!("#        finishing variablization");
                match mcts.revisions[*rh].spec {
                    Some(MCTSMoveState::Variablize(ref vs, ref types)) => {
                        let mut types = types.clone();
                        let mut clauses = rs.clone();
                        let mut progress = *n;
                        if progress.is_none() {
                            let mut order_first = (0..vs.len()).collect_vec();
                            order_first.shuffle(rng);
                            for first in order_first {
                                let (m, tp, places) = &vs[first];
                                if let Some(rule) =
                                    trs.apply_variablization(&tp, &places, &clauses[*m], &mut types)
                                {
                                    clauses[*m] = rule;
                                    if let Some(new_trs) = trs.adopt_solution(&mut clauses.clone())
                                    {
                                        trs = new_trs;
                                        progress.replace(first + 1);
                                        break;
                                    }
                                }
                            }
                        }
                        for (m, tp, places) in vs.iter().skip(progress?) {
                            if rng.gen() {
                                if let Some(rule) =
                                    trs.apply_variablization(&tp, &places, &clauses[*m], &mut types)
                                {
                                    clauses[*m] = rule;
                                    if let Some(new_trs) = trs.adopt_solution(&mut clauses.clone())
                                    {
                                        trs = new_trs;
                                    }
                                }
                            }
                        }
                        println!("#        success");
                        *steps_remaining = steps_remaining.saturating_sub(1);
                        return Some(trs);
                    }
                    _ => panic!("Variablize cannot find reference data"),
                }
            }
            Some(MCTSMoveState::MemorizeData(progress)) => {
                println!("#         finishing memorization");
                let mut lower_bound = *progress;
                if progress.is_none() {
                    let mut order_first = (0..mcts.data.len()).collect_vec();
                    order_first.shuffle(rng);
                    for first in order_first {
                        if trs.append_clauses(vec![mcts.data[first].clone()]).is_ok() {
                            lower_bound.replace(first + 1);
                            break;
                        }
                    }
                }
                // We don't mind failure now that we have one success.
                for rule in mcts.data.iter().skip(lower_bound?) {
                    if rng.gen() {
                        trs.append_clauses(vec![rule.clone()]).ok();
                    }
                }
                println!("#        success");
                *steps_remaining = steps_remaining.saturating_sub(1);
                return Some(trs);
            }
            Some(MCTSMoveState::RegenerateRule(progress)) => {
                let mut idxs = match progress {
                    Some((n, _)) => vec![*n],
                    None => (0..trs.len()).collect_vec(),
                };
                idxs.shuffle(rng);
                for idx in idxs {
                    let mut contexts = match progress {
                        Some((_, c)) => vec![c.clone()],
                        None => {
                            let rulecontext = RuleContext::from(trs.utrs.rules[idx].clone());
                            rulecontext
                                .subcontexts()
                                .into_iter()
                                .filter_map(|(_, place)| rulecontext.replace(&place, Context::Hole))
                                .filter(|rc| RuleContext::is_valid(&rc.lhs, &rc.rhs))
                                .collect_vec()
                        }
                    };
                    contexts.shuffle(rng);
                    for mut context in contexts {
                        context.canonicalize(&mut HashMap::new());
                        println!(
                            "#         finishing regeneration with: {}",
                            context.pretty(trs.lex.signature())
                        );
                        for _ in 0..100 {
                            println!("#           looping");
                            if let Ok(rule) = trs.lex.sample_rule_from_context(
                                &context,
                                mcts.params.atom_weights,
                                GenerationLimit::TotalSize(
                                    context.size() + mcts.params.max_size - 1,
                                ),
                                mcts.params.invent,
                                &mut trs.lex.0.ctx.clone(),
                                rng,
                            ) {
                                let old_rule = trs.utrs.rules[idx].clone();
                                let mut new_trs = trs.clone();
                                if new_trs.utrs.replace(idx, &old_rule, rule.clone()).is_ok() {
                                    println!(
                                        "#           success: {}",
                                        rule.pretty(&new_trs.lex.signature())
                                    );
                                    *steps_remaining = steps_remaining.saturating_sub(1);
                                    return Some(new_trs);
                                }
                            }
                        }
                    }
                }
            }
            Some(MCTSMoveState::SampleRule(ref context)) => {
                println!(
                    "#         finishing sample: {}",
                    context.pretty(&trs.lex.signature())
                );
                for _ in 0..100 {
                    println!("#           looping");
                    if let Ok(rule) = trs.lex.sample_rule_from_context(
                        context,
                        mcts.params.atom_weights,
                        GenerationLimit::TermSize(mcts.params.max_size),
                        mcts.params.invent,
                        &mut trs.lex.0.ctx.clone(),
                        rng,
                    ) {
                        if trs.append_clauses(vec![rule.clone()]).is_ok() {
                            println!("#           success: {}", rule.pretty(&trs.lex.signature()));
                            *steps_remaining = steps_remaining.saturating_sub(1);
                            return Some(trs);
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
        let mut moves = vec![];
        match self.handle {
            StateHandle::Terminal(..) => (),
            StateHandle::Revision(rh) => mcts.revisions[rh].available_moves(mcts, &mut moves, rh),
        }
        moves
    }
    fn make_move<R: Rng>(
        &self,
        mv: &Self::Move,
        mcts: &mut TRSMCTS<'a, 'b>,
        _rng: &mut R,
    ) -> Option<(Self, Option<Self::AbstractDepthAdjustment>)> {
        let state = match self.handle {
            StateHandle::Terminal(..) => panic!("cannot move from terminal"),
            StateHandle::Revision(rh) => mcts.make_move(mv, rh),
        }?;
        Some(mcts.add_state(state))
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
                let hh = mcts.revisions[rh].trs;
                let trs = &mcts.hypotheses[hh].trs;
                Value::String(mv.pretty(&trs.lex, &mcts.data))
            }
        }
    }
    fn describe_self(&self, mcts: &TRSMCTS) -> Value {
        match self.handle {
            StateHandle::Terminal(th) => {
                let hh = mcts.terminals[th].trs;
                let trs = &mcts.hypotheses[hh].trs;
                let trs_string = trs.utrs.pretty(trs.lex.signature());
                json!({
                    "type": "terminal",
                    "trs": trs_string,
                })
            }
            StateHandle::Revision(rh) => {
                let hh = mcts.revisions[rh].trs;
                let trs = &mcts.hypotheses[hh].trs;
                let trs_string = trs.utrs.pretty(trs.lex.signature());
                let playout_string = match mcts.revisions[rh].playout {
                    PlayoutState::Failed => "failed".to_string(),
                    PlayoutState::Untried => "untried".to_string(),
                    PlayoutState::Success(hh) => {
                        let playout = &mcts.hypotheses[hh].trs;
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
        data: &'a [Rule],
        input: Option<&'a Term>,
        model: ModelParams,
        params: MCTSParams,
    ) -> TRSMCTS<'a, 'b> {
        TRSMCTS {
            lexicon,
            bg,
            deterministic,
            data,
            input,
            model,
            params,
            hypotheses: vec![],
            terminals: vec![],
            revisions: vec![],
            best: std::f64::NEG_INFINITY,
        }
    }
    fn make_move(&mut self, mv: &MCTSMove, rh: RevisionHandle) -> Option<StateKind> {
        // TODO: remove need for clone
        let revision = self.revisions[rh].clone();
        revision.make_move(mv, self, rh)
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
    pub fn find_hypothesis(&mut self, mut trs: TRS<'a, 'b>) -> HypothesisHandle {
        trs.utrs.canonicalize(&mut HashMap::new());
        match self
            .hypotheses
            .iter()
            .position(|h| TRS::same_shape(&h.trs, &trs))
        {
            Some(hh) => hh,
            None => {
                self.hypotheses.push(Hypothesis::new(
                    trs, &self.data, self.input, 1.0, self.model,
                ));
                self.hypotheses.len() - 1
            }
        }
    }
    pub fn root(&mut self) -> MCTSState {
        let mut trs = TRS::new_unchecked(&self.lexicon, self.deterministic, self.bg, vec![]);
        trs.identify_symbols();
        let state = Revision {
            trs: self.find_hypothesis(trs),
            spec: None,
            n: 0,
            playout: PlayoutState::Untried,
        };
        self.add_revision(state).0
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
            childless.len() + 1,
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
                PlayoutState::Untried => panic!("shouldn't reread empty state"),
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
                        .trs
                        .to_string()
                        .lines()
                        .join(" ")
                );
                mcts.hypotheses[mcts.terminals[th].trs].lposterior
            }
            StateHandle::Revision(rh) => match mcts.revisions[rh].playout {
                PlayoutState::Untried => {
                    println!(
                        "#       playing out {}",
                        mcts.hypotheses[mcts.revisions[rh].trs]
                            .trs
                            .to_string()
                            .lines()
                            .join(" ")
                    );
                    if let Some(trs) = mcts.revisions[rh].playout(mcts, rng) {
                        println!(
                            "#         simulated: \"{}\"",
                            trs.to_string().lines().join(" ")
                        );
                        let hh = mcts.find_hypothesis(trs);
                        mcts.revisions[rh].playout = PlayoutState::Success(hh);
                        mcts.hypotheses[hh].lposterior
                    } else {
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

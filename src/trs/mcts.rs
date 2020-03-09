// TODO:
// - Update posterior computation to be more efficient. Let hypothesis store
//   likelihoods so just incrementally update.
// - add revision moves
use itertools::Itertools;
use mcts::{MoveEvaluator, MoveInfo, NodeHandle, SearchTree, State, StateEvaluator, MCTS};
use polytype::TypeSchema;
use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng,
};
use std::collections::HashMap;
use term_rewriting::{Atom, Context, Rule, RuleContext};
use trs::{Hypothesis, Lexicon, ModelParams, TRS};
use utils::logsumexp;

type SamplingHandle = usize;
type RevisingHandle = usize;
type TerminalHandle = usize;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum StateHandle {
    Sampling(SamplingHandle),
    Revising(RevisingHandle),
    Terminal(TerminalHandle),
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct MCTSState {
    handle: StateHandle,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Sample<'b> {
    lex: Lexicon<'b>,
    rules: Vec<Rule>,
    spec: Option<MCTSMoveState>,
    data: bool,
    playout: Option<TerminalHandle>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Revise<'a, 'b> {
    n: usize,
    trs: TRS<'a, 'b>,
    spec: Option<MCTSMoveState>,
}

pub enum StateKind<'a, 'b> {
    Sampling(Sample<'b>),
    Terminal(Hypothesis<'a, 'b>),
    Revising(Revise<'a, 'b>),
}

pub struct TRSMCTS<'a, 'b> {
    pub lexicon: Lexicon<'b>,
    pub bg: &'a [Rule],
    pub deterministic: bool,
    pub data: &'a [Rule],
    pub samplings: Vec<Sample<'b>>,
    pub revisings: Vec<Revise<'a, 'b>>,
    pub terminals: Vec<Hypothesis<'a, 'b>>,
    pub max_revisions: usize,
    pub params: ModelParams,
    pub max_depth: usize,
    pub max_states: usize,
    pub atom_weights: (f64, f64, f64, f64),
    pub invent: bool,
    pub max_size: usize,
}

pub struct MCTSMoveEvaluator;

pub struct MCTSStateEvaluator;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum MCTSMoveState {
    MemorizeData(Option<usize>),
    SampleRule(RuleContext),
    RegenerateRule(Option<(usize, RuleContext)>),
    DeleteRules(Option<usize>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MCTSMove {
    MemorizeData,
    SampleRule,
    RegenerateRule,
    RegenerateThisRule(usize, RuleContext),
    DeleteRules,
    Empty,
    MemorizeDatum(Option<usize>),
    SampleAtom(Option<Atom>),
    DeleteRule(Option<usize>),
    // TODO: the revision moves
    // - Generalize
    // - Compose
    // - Recurse
    // - Variablize
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Move {
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
    Compose,
}

impl<'a, 'b> State<TRSMCTS<'a, 'b>> for MCTSState {
    type Move = MCTSMove;
    type MoveList = Vec<Self::Move>;
    fn available_moves(&self, mcts: &mut TRSMCTS) -> Self::MoveList {
        let mut moves = vec![];
        match self.handle {
            StateHandle::Terminal(..) => (),
            StateHandle::Revising(rh) => mcts.revisings[rh].available_moves(mcts, &mut moves),
            StateHandle::Sampling(sh) => mcts.samplings[sh].available_moves(mcts, &mut moves),
        }
        moves
    }
    fn make_move<R: Rng>(&self, mv: &Self::Move, mcts: &mut TRSMCTS<'a, 'b>, rng: &mut R) -> Self {
        let state = match self.handle {
            StateHandle::Terminal(..) => panic!("inconsistent state: no move from terminal"),
            StateHandle::Revising(..) => panic!("not ready for this"),
            StateHandle::Sampling(sh) => mcts.samplings[sh].make_move(mv, mcts, rng),
        };
        mcts.add_state(state)
    }
    fn add_moves_for_new_data(&self, moves: &[Self::Move], mcts: &mut TRSMCTS) -> Vec<Self::Move> {
        self.available_moves(mcts)
            .into_iter()
            .filter(|m| !moves.contains(&m))
            .collect()
    }
}

impl<'a, 'b> Revise<'a, 'b> {
    pub fn available_moves(&self, mcts: &TRSMCTS, moves: &mut Vec<MCTSMove>) {
        match self.spec {
            None => {
                // A TRS can always sample a new rule.
                moves.push(MCTSMove::SampleRule);
                // You must have a rule in order to regenerate a rule.
                if !self.trs.is_empty() {
                    moves.push(MCTSMove::RegenerateRule);
                }
                // A TRS must always have at least one rule.
                if self.trs.len() > 1 {
                    moves.push(MCTSMove::DeleteRules);
                }
                // We can only add data if there's data to add.
                if !mcts.data.is_empty() {
                    moves.push(MCTSMove::MemorizeData);
                }
            }
            Some(MCTSMoveState::MemorizeData(n)) => {
                let lower_bound = n.unwrap_or(0);
                (lower_bound..mcts.data.len())
                    .map(|i_datum| MCTSMove::MemorizeDatum(Some(i_datum)))
                    .for_each(|mv| moves.push(mv));
                if n.is_some() {
                    moves.push(MCTSMove::MemorizeDatum(None));
                }
            }
            Some(MCTSMoveState::DeleteRules(n)) => {
                let lower_bound = n.unwrap_or(0);
                (lower_bound..self.trs.len())
                    .map(|rule| MCTSMove::DeleteRule(Some(rule)))
                    .for_each(|mv| moves.push(mv));
                if n.is_some() {
                    moves.push(MCTSMove::DeleteRule(None));
                }
            }
            Some(MCTSMoveState::SampleRule(ref context))
            | Some(MCTSMoveState::RegenerateRule(Some((_, ref context)))) => {
                if let Some(place) = context.leftmost_hole() {
                    self.trs
                        .lex
                        .rulecontext_fillers(&context, &place)
                        .into_iter()
                        .map(|atom| MCTSMove::SampleAtom(atom))
                        .for_each(|mv| moves.push(mv))
                }
            }
            Some(MCTSMoveState::RegenerateRule(None)) => {
                for (i, rule) in self.trs.utrs.rules.iter().enumerate() {
                    let rulecontext = RuleContext::from(rule.clone());
                    for (_, place) in rulecontext.subcontexts() {
                        let context = rulecontext.replace(&place, Context::Hole).unwrap();
                        moves.push(MCTSMove::RegenerateThisRule(i, context));
                    }
                }
            }
        }
    }
    pub fn make_move<R: Rng>(
        &self,
        mv: &MCTSMove,
        mcts: &TRSMCTS<'a, 'b>,
        _rng: &mut R,
    ) -> StateKind<'a, 'b> {
        match *mv {
            MCTSMove::DeleteRules => {
                // We're stating an intention: just the internal state changes.
                let mut state = self.clone();
                state.spec = Some(MCTSMoveState::DeleteRules(None));
                StateKind::Revising(state)
            }
            MCTSMove::DeleteRule(None) => {
                // You're done deleting: the rules don't change, but internal state does.
                if self.n + 1 >= mcts.max_revisions {
                    let hypothesis =
                        Hypothesis::new(self.trs.clone(), &mcts.data, 1.0, mcts.params);
                    StateKind::Terminal(hypothesis)
                } else {
                    let mut state = self.clone();
                    state.spec = None;
                    state.n += 1;
                    StateKind::Revising(state)
                }
            }
            MCTSMove::DeleteRule(Some(n)) => {
                // You're actively deleting or finished not by choice: rules and state change.
                let mut trs = self.trs.clone();
                trs.utrs.remove_idx(n).ok();
                if n >= trs.len() {
                    if self.n + 1 >= mcts.max_revisions {
                        let hypothesis =
                            Hypothesis::new(self.trs.clone(), &mcts.data, 1.0, mcts.params);
                        StateKind::Terminal(hypothesis)
                    } else {
                        let revising = Revise {
                            spec: None,
                            n: self.n + 1,
                            trs: trs,
                        };
                        StateKind::Revising(revising)
                    }
                } else {
                    let revising = Revise {
                        spec: Some(MCTSMoveState::DeleteRules(Some(n))),
                        n: self.n,
                        trs: trs,
                    };
                    StateKind::Revising(revising)
                }
            }
            MCTSMove::MemorizeData => {
                let mut state = self.clone();
                state.spec = Some(MCTSMoveState::MemorizeData(None));
                StateKind::Revising(state)
            }
            MCTSMove::MemorizeDatum(None) => {
                // You're done memorizing: the rules don't change, but internal state does.
                if self.n + 1 >= mcts.max_revisions {
                    let hypothesis =
                        Hypothesis::new(self.trs.clone(), &mcts.data, 1.0, mcts.params);
                    StateKind::Terminal(hypothesis)
                } else {
                    let mut state = self.clone();
                    state.spec = None;
                    state.n += 1;
                    StateKind::Revising(state)
                }
            }
            MCTSMove::MemorizeDatum(Some(n)) => {
                // You're actively memorizing: rules and state change.
                let mut trs = self.trs.clone();
                trs.append_clauses(vec![mcts.data[n].clone()]).ok();
                let revising = Revise {
                    spec: if n + 1 == mcts.data.len() {
                        None
                    } else {
                        Some(MCTSMoveState::MemorizeData(Some(n)))
                    },
                    n: self.n,
                    trs: trs,
                };
                StateKind::Revising(revising)
            }
            MCTSMove::SampleRule => {
                // You're stating the intention to memorize: state changes.
                let mut state = self.clone();
                state.spec = Some(MCTSMoveState::SampleRule(RuleContext::default()));
                StateKind::Revising(state)
            }
            MCTSMove::SampleAtom(atom) => match self.spec {
                Some(MCTSMoveState::SampleRule(ref rc)) => {
                    let place = rc.leftmost_hole().expect("Sampling rule without holes.");
                    let mut state = self.clone();
                    let tp = {
                        let mut types = HashMap::new();
                        state.trs.lex.infer_rulecontext(rc, &mut types).drop().ok();
                        let lex_vars = state.trs.lex.free_vars_applied();
                        let schema = types[&place].generalize(&lex_vars);
                        state.trs.lex.instantiate(&schema)
                    };
                    let atom =
                        atom.unwrap_or_else(|| Atom::Variable(state.trs.lex.invent_variable(&tp)));
                    if let Some(new_context) = rc.replace(&place, Context::from(atom)) {
                        if let Ok(rule) = new_context.to_rule() {
                            state.trs.append_clauses(vec![rule]).ok();
                            state.spec = None;
                        } else {
                            state.spec = Some(MCTSMoveState::SampleRule(new_context));
                        }
                        println!("#   {}", state.trs.to_string().lines().join(" "));
                        StateKind::Revising(state)
                    } else {
                        panic!("RuleContext::replace failed");
                    }
                }
                Some(MCTSMoveState::RegenerateRule(Some((n, ref context)))) => {
                    let place = context.leftmost_hole().expect("rule has no holes.");
                    let mut state = self.clone();
                    let tp = {
                        let mut types = HashMap::new();
                        state
                            .trs
                            .lex
                            .infer_rulecontext(context, &mut types)
                            .drop()
                            .ok();
                        let lex_vars = state.trs.lex.free_vars_applied();
                        let schema = types[&place].generalize(&lex_vars);
                        state.trs.lex.instantiate(&schema)
                    };
                    let atom =
                        atom.unwrap_or_else(|| Atom::Variable(state.trs.lex.invent_variable(&tp)));
                    if let Some(new_context) = context.replace(&place, Context::from(atom)) {
                        if let Ok(rule) = new_context.to_rule() {
                            state.trs.utrs.rules.push(rule);
                            state.trs.utrs.rules.swap_remove(n);
                            state.spec = None;
                        } else {
                            state.spec =
                                Some(MCTSMoveState::RegenerateRule(Some((n, new_context))));
                        }
                        println!("#   {}", state.trs.to_string().lines().join(" "));
                        StateKind::Revising(state)
                    } else {
                        panic!("RuleContext::replace failed");
                    }
                }
                _ => panic!("move state doesn't match move"),
            },
            MCTSMove::RegenerateRule => {
                // You're stating the intention to regenerate: state changes.
                let mut state = self.clone();
                state.spec = Some(MCTSMoveState::RegenerateRule(None));
                StateKind::Revising(state)
            }
            MCTSMove::RegenerateThisRule(n, ref context) => {
                // You're stating where you want to regenerate: state changes.
                let mut state = self.clone();
                state.spec = Some(MCTSMoveState::RegenerateRule(Some((n, context.clone()))));
                StateKind::Revising(state)
            }
            _ => panic!("Invalid move"),
        }
    }
}

impl<'b> Sample<'b> {
    pub fn available_moves(&self, mcts: &TRSMCTS, moves: &mut Vec<MCTSMove>) {
        match self.spec {
            None => {
                // There must be at least one rule in your TRS.
                if !self.rules.is_empty() {
                    moves.push(MCTSMove::Empty);
                }
                // We can always sample
                moves.push(MCTSMove::SampleRule);
                // We can only add data once and that only if there's data.
                if !(self.data || mcts.data.is_empty()) {
                    moves.push(MCTSMove::MemorizeData);
                }
            }
            Some(MCTSMoveState::MemorizeData(None)) => {
                (0..mcts.data.len())
                    .map(|i_datum| MCTSMove::MemorizeDatum(Some(i_datum)))
                    .for_each(|mv| moves.push(mv));
            }
            Some(MCTSMoveState::MemorizeData(Some(n))) => {
                (n + 1..mcts.data.len())
                    .map(|i_datum| MCTSMove::MemorizeDatum(Some(i_datum)))
                    .for_each(|mv| moves.push(mv));
                moves.push(MCTSMove::MemorizeDatum(None));
            }
            Some(MCTSMoveState::SampleRule(ref context)) => {
                if let Some(place) = context.leftmost_hole() {
                    self.lex
                        .rulecontext_fillers(&context, &place)
                        .into_iter()
                        .map(|atom| MCTSMove::SampleAtom(atom))
                        .for_each(|mv| moves.push(mv))
                }
            }
            _ => panic!("inconsistent state: invalid move state"),
        }
    }
    pub fn make_move<'a, R: Rng>(
        &self,
        mv: &MCTSMove,
        mcts: &TRSMCTS<'a, 'b>,
        _rng: &mut R,
    ) -> StateKind<'a, 'b> {
        match *mv {
            MCTSMove::Empty => {
                let trs =
                    TRS::new_unchecked(&self.lex, mcts.deterministic, mcts.bg, self.rules.clone());
                let revising = Revise {
                    n: 0,
                    spec: None,
                    trs: trs,
                };
                StateKind::Revising(revising)
            }
            MCTSMove::MemorizeData => {
                let mut state = self.clone();
                state.spec = Some(MCTSMoveState::MemorizeData(None));
                state.playout = None;
                StateKind::Sampling(state)
            }
            MCTSMove::MemorizeDatum(None) => {
                // You're done memorizing: the rules don't change, but internal state does.
                let mut state = self.clone();
                state.spec = None;
                state.playout = None;
                state.data = true;
                StateKind::Sampling(state)
            }
            MCTSMove::MemorizeDatum(Some(n)) => {
                // You're actively memorizing: rules and state change.
                let mut state = self.clone();
                state.rules.push(mcts.data[n].clone());
                state.spec = if n + 1 == mcts.data.len() {
                    None
                } else {
                    Some(MCTSMoveState::MemorizeData(Some(n)))
                };
                state.playout = None;
                StateKind::Sampling(state)
            }
            MCTSMove::SampleRule => {
                // You're stating the intention to memorize: state changes.
                let mut state = self.clone();
                state.spec = Some(MCTSMoveState::SampleRule(RuleContext::default()));
                state.playout = None;
                StateKind::Sampling(state)
            }
            MCTSMove::SampleAtom(atom) => match self.spec {
                Some(MCTSMoveState::SampleRule(ref rc)) => {
                    let place = rc.leftmost_hole().expect("How did this move happen?");
                    let mut state = self.clone();
                    let tp = {
                        let mut types = HashMap::new();
                        state.lex.infer_rulecontext(rc, &mut types).drop().ok();
                        let lex_vars = state.lex.free_vars_applied();
                        let schema = types[&place].generalize(&lex_vars);
                        state.lex.instantiate(&schema)
                    };
                    let atom =
                        atom.unwrap_or_else(|| Atom::Variable(state.lex.invent_variable(&tp)));
                    if let Some(new_context) = rc.replace(&place, Context::from(atom)) {
                        if let Ok(rule) = new_context.to_rule() {
                            state.rules.push(rule);
                            state.spec = None;
                            state.playout = None;
                            println!(
                                "#   {}",
                                state
                                    .rules
                                    .iter()
                                    .map(|r| r.pretty(&state.lex.signature()))
                                    .join("; ")
                            );
                            StateKind::Sampling(state)
                        } else {
                            let mut state = self.clone();
                            println!(
                                "#   {}{}{};",
                                state
                                    .rules
                                    .iter()
                                    .map(|r| r.pretty(&state.lex.signature()))
                                    .join("; "),
                                if state.rules.is_empty() { "" } else { "; " },
                                new_context.pretty(&state.lex.signature())
                            );
                            state.spec = Some(MCTSMoveState::SampleRule(new_context));
                            state.playout = None;
                            StateKind::Sampling(state)
                        }
                    } else {
                        panic!("inconsistent state: couldn't fill a supposed hole");
                    }
                }
                _ => panic!("inconsistent state: invalid move"),
            },
            _ => panic!("inconsistent state: invalid move"),
        }
    }
    pub fn playout<'a, R: Rng>(&self, mcts: &TRSMCTS<'a, 'b>, rng: &mut R) -> TRS<'a, 'b> {
        let mut data = self.data;
        let mut lex = self.lex.clone();
        let mut rules = self.rules.clone();
        match self.spec {
            None => (),
            Some(MCTSMoveState::MemorizeData(progress)) => {
                let lower_bound = progress.unwrap_or(0);
                for rule in mcts.data.iter().skip(lower_bound) {
                    if rng.gen() {
                        rules.push(rule.clone());
                    }
                }
                data = true;
            }
            Some(MCTSMoveState::SampleRule(ref context)) => {
                println!(
                    "#         finalizing rule context: {}",
                    context.pretty(&lex.signature())
                );
                loop {
                    println!("#           looping");
                    if let Ok(rule) = lex
                        .sample_rule_from_context(
                            context.clone(),
                            mcts.atom_weights,
                            mcts.invent,
                            mcts.max_size,
                            rng,
                        )
                        .drop()
                    {
                        println!("#         sampled: {}", rule.pretty(&lex.signature()));
                        rules.push(rule);
                        break;
                    }
                }
            }
            _ => panic!("inconsistent state: invalid move selected"),
        }
        loop {
            // data, new_rule, empty
            let moves =
                WeightedIndex::new(&[(!(data || mcts.data.is_empty()) as usize as f64), 1.0, 1.0])
                    .unwrap();
            match moves.sample(rng) {
                0 => {
                    println!("#         adding data");
                    for rule in mcts.data {
                        if rng.gen() {
                            rules.push(rule.clone());
                        }
                    }
                    data = true;
                }
                1 => {
                    let schema = TypeSchema::Monotype(lex.fresh_type_variable());
                    println!("#         sampling a rule");
                    loop {
                        println!("#           looping");
                        if let Ok(rule) = lex
                            .sample_rule(
                                &schema,
                                mcts.atom_weights,
                                mcts.invent,
                                mcts.max_size,
                                rng,
                            )
                            .drop()
                        {
                            println!("#           sampled: {}", rule.pretty(&lex.signature()));
                            rules.push(rule);
                            break;
                        }
                    }
                }
                2 => {
                    println!("#         stopping");
                    break;
                }
                _ => unreachable!(),
            }
        }
        TRS::new_unchecked(&lex, mcts.deterministic, mcts.bg, rules)
    }
}

impl<'a, 'b> MCTS for TRSMCTS<'a, 'b> {
    type StateEval = MCTSStateEvaluator;
    type MoveEval = MCTSMoveEvaluator;
    type State = MCTSState;
    fn max_depth(&self) -> usize {
        self.max_depth
    }
    fn max_states(&self) -> usize {
        self.max_states
    }
    fn combine_qs(&self, q1: f64, q2: f64) -> f64 {
        logsumexp(&[q1, q2])
    }
}

impl<'a, 'b> TRSMCTS<'a, 'b> {
    pub fn new(
        lexicon: Lexicon<'b>,
        bg: &'a [Rule],
        deterministic: bool,
        data: &'a [Rule],
        params: ModelParams,
        max_depth: usize,
        max_states: usize,
        max_revisions: usize,
        invent: bool,
        atom_weights: (f64, f64, f64, f64),
        max_size: usize,
    ) -> TRSMCTS<'a, 'b> {
        TRSMCTS {
            lexicon,
            bg,
            deterministic,
            data,
            max_revisions,
            max_depth,
            max_states,
            params,
            invent,
            atom_weights,
            max_size,
            terminals: vec![],
            samplings: vec![],
            revisings: vec![],
        }
    }
    pub fn add_state(&mut self, state: StateKind<'a, 'b>) -> MCTSState {
        match state {
            StateKind::Terminal(h) => self.add_terminal(h),
            StateKind::Revising(r) => self.add_revising(r),
            StateKind::Sampling(s) => self.add_sampling(s),
        }
    }
    pub fn add_sampling(&mut self, state: Sample<'b>) -> MCTSState {
        self.samplings.push(state);
        let handle = StateHandle::Sampling(self.samplings.len() - 1);
        MCTSState { handle }
    }
    pub fn add_revising(&mut self, state: Revise<'a, 'b>) -> MCTSState {
        self.revisings.push(state);
        let handle = StateHandle::Revising(self.revisings.len() - 1);
        MCTSState { handle }
    }
    pub fn add_terminal(&mut self, state: Hypothesis<'a, 'b>) -> MCTSState {
        self.terminals.push(state);
        let handle = StateHandle::Terminal(self.terminals.len() - 1);
        MCTSState { handle }
    }
    pub fn root(&mut self) -> MCTSState {
        let state = Sample {
            lex: self.lexicon.clone(),
            rules: vec![],
            spec: None,
            data: false,
            playout: None,
        };
        self.add_sampling(state)
    }
}

impl<'a, 'b> MoveEvaluator<TRSMCTS<'a, 'b>> for MCTSMoveEvaluator {
    type MoveEvaluation = f64;
    fn choose<'c, MoveIter>(
        &self,
        moves: MoveIter,
        nh: NodeHandle,
        tree: &SearchTree<TRSMCTS<'a, 'b>>,
    ) -> Option<&'c MoveInfo<TRSMCTS<'a, 'b>>>
    where
        MoveIter: Iterator<Item = &'c MoveInfo<TRSMCTS<'a, 'b>>>,
    {
        // Split the moves into those with and without children.
        let (childful, mut childless): (Vec<_>, Vec<_>) = moves.partition(|mv| mv.child.is_some());
        // Take the first childless move, or perform UCT on childed moves.
        if let Some(mv) = childless.pop() {
            println!("#   There are childless. We chose: {:?}.", mv.mov);
            Some(mv)
        } else {
            childful
                .into_iter()
                .map(|mv| {
                    let ch = mv.child.expect("INVARIANT: partition failed us");
                    let node = tree.node(nh);
                    let child = tree.node(ch);
                    println!(
                        "#     UCT: {}'s q/n: {:.3} / {:.3} + sqrt(ln({:.3}) / {:.3})",
                        ch,
                        child.q.exp(),
                        child.n,
                        node.n,
                        child.n
                    );
                    let score = child.q.exp() / child.n + (node.n.ln() / child.n).sqrt();
                    (mv, score)
                })
                .max_by(|x, y| x.1.partial_cmp(&y.1).expect("There a NaN on the loose!"))
                .map(|(mv, _)| {
                    println!("#     we're going with {:?}", mv.mov);
                    mv
                })
                .or_else(|| {
                    println!("#     no available moves");
                    None
                })
        }
    }
}

impl<'a, 'b> StateEvaluator<TRSMCTS<'a, 'b>> for MCTSStateEvaluator {
    type StateEvaluation = f64;
    fn zero(&self) -> f64 {
        std::f64::NEG_INFINITY
    }
    fn evaluate<R: Rng>(
        &self,
        state: &<TRSMCTS<'a, 'b> as MCTS>::State,
        mcts: &mut TRSMCTS<'a, 'b>,
        rng: &mut R,
    ) -> Self::StateEvaluation {
        println!("#     evaluating");
        match state.handle {
            StateHandle::Terminal(sh) => {
                println!("#       node is terminal");
                mcts.terminals[sh].lposterior
            }
            StateHandle::Revising(_) => panic!("not prepared for this"),
            StateHandle::Sampling(sh) => match &mcts.samplings[sh].playout {
                Some(th) => {
                    println!("#       found a playout");
                    mcts.terminals[*th].lposterior
                }
                None => {
                    println!("#       playing out");
                    let trs = mcts.samplings[sh].playout(mcts, rng);
                    println!(
                        "#         simulated: \"{}\"",
                        trs.to_string().lines().join(" ")
                    );
                    let h = Hypothesis::new(trs, &mcts.data, 1.0, mcts.params);
                    let score = h.lposterior;
                    let th = mcts.terminals.len();
                    mcts.terminals.push(h);
                    mcts.samplings[sh].playout = Some(th);
                    score
                }
            },
        }
    }
}

//impl Move {
//    pub fn take<'a, 'b, R: Rng>(
//        &self,
//        mcts: &TRSMCTS<'a, 'b>,
//        rng: &mut R,
//        parents: &[&TRS<'a, 'b>],
//    ) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
//        match *self {
//            Move::Compose => parents[0].compose(),
//            Move::DeleteRule => parents[0].delete_rule(),
//            Move::DeleteRules(t) => parents[0].delete_rules(rng, t),
//            Move::Generalize => parents[0].generalize(),
//            Move::LocalDifference => parents[0].local_difference(rng),
//            Move::Memorize(deterministic) => Ok(TRS::memorize(
//                &mcts.lexicon,
//                deterministic,
//                &mcts.bg,
//                &mcts.data,
//            )),
//            Move::MemorizeOne => parents[0].memorize_one(&mcts.data),
//            Move::Recurse(n) => parents[0].recurse(n),
//            Move::RegenerateRule(aw, mss) => parents[0].regenerate_rule(aw, mss, rng),
//            Move::SampleRule(aw, mss) => parents[0].sample_rule(aw, mss, rng),
//            Move::Variablize => parents[0].variablize(),
//        }
//    }
//    pub fn deterministic(&self) -> bool {
//        match *self {
//            Move::Compose
//            | Move::DeleteRule
//            | Move::Generalize
//            | Move::LocalDifference
//            | Move::Memorize(_)
//            | Move::MemorizeOne
//            | Move::Variablize => true,
//            Move::DeleteRules(_)
//            | Move::Recurse(_)
//            | Move::RegenerateRule(..)
//            | Move::SampleRule(..) => false,
//        }
//    }
//    pub fn num_parents(&self) -> usize {
//        match *self {
//            Move::Memorize(_) => 0,
//            _ => 1,
//        }
//    }
//    pub fn applies(&self, parents: &[&TRS], data: &[Rule]) -> bool {
//        self.num_parents() == parents.len()
//            && match *self {
//                Move::Compose
//                | Move::DeleteRule
//                | Move::DeleteRules(_)
//                | Move::Generalize
//                | Move::LocalDifference
//                | Move::Recurse(_)
//                | Move::RegenerateRule(..)
//                | Move::Variablize => !parents[0].is_empty(),
//                Move::MemorizeOne => !data.is_empty(),
//                _ => true,
//            }
//    }
//    pub fn data_sensitive(&self) -> bool {
//        match *self {
//            Move::Memorize(_) | Move::MemorizeOne => true,
//            _ => false,
//        }
//    }
//}

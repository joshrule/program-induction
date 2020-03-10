// TODO:
// - Update posterior computation to be more efficient. Let hypothesis store
//   likelihoods so just incrementally update.
// - add revision moves
use itertools::Itertools;
use mcts::{MoveEvaluator, MoveInfo, NodeHandle, SearchTree, State, StateEvaluator, MCTS};
use polytype::TypeSchema;
use rand::{
    distributions::{Distribution, WeightedIndex},
    prelude::IteratorRandom,
    Rng,
};
use std::collections::HashMap;
use term_rewriting::{Atom, Context, Rule, RuleContext};
use trs::{Hypothesis, Lexicon, ModelParams, TRS};
use utils::logsumexp;

type RevisionHandle = usize;
type TerminalHandle = usize;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum StateHandle {
    Revision(RevisionHandle),
    Terminal(TerminalHandle),
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct MCTSState {
    handle: StateHandle,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Revise<'a, 'b> {
    n: usize,
    trs: TRS<'a, 'b>,
    spec: Option<MCTSMoveState>,
    playout: Option<TerminalHandle>,
}

pub enum StateKind<'a, 'b> {
    Terminal(Hypothesis<'a, 'b>),
    Revision(Revise<'a, 'b>),
}

pub struct TRSMCTS<'a, 'b> {
    pub lexicon: Lexicon<'b>,
    pub bg: &'a [Rule],
    pub deterministic: bool,
    pub data: &'a [Rule],
    pub revisions: Vec<Revise<'a, 'b>>,
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
    DeleteRules,
    RegenerateThisRule(usize, RuleContext),
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
            StateHandle::Revision(rh) => mcts.revisions[rh].available_moves(mcts, &mut moves),
        }
        moves
    }
    fn make_move<R: Rng>(&self, mv: &Self::Move, mcts: &mut TRSMCTS<'a, 'b>, rng: &mut R) -> Self {
        let state = match self.handle {
            StateHandle::Terminal(..) => panic!("inconsistent state: no move from terminal"),
            StateHandle::Revision(rh) => mcts.revisions[rh].make_move(mv, mcts, rng),
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
                state.playout = None;
                StateKind::Revision(state)
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
                    state.playout = None;
                    StateKind::Revision(state)
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
                            playout: None,
                        };
                        StateKind::Revision(revising)
                    }
                } else {
                    let revising = Revise {
                        spec: Some(MCTSMoveState::DeleteRules(Some(n))),
                        n: self.n,
                        trs: trs,
                        playout: None,
                    };
                    StateKind::Revision(revising)
                }
            }
            MCTSMove::MemorizeData => {
                let mut state = self.clone();
                state.spec = Some(MCTSMoveState::MemorizeData(None));
                state.playout = None;
                StateKind::Revision(state)
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
                    StateKind::Revision(state)
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
                    playout: None,
                };
                StateKind::Revision(revising)
            }
            MCTSMove::SampleRule => {
                // You're stating the intention to memorize: state changes.
                let mut state = self.clone();
                state.spec = Some(MCTSMoveState::SampleRule(RuleContext::default()));
                state.playout = None;
                StateKind::Revision(state)
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
                        state.playout = None;
                        println!("#   \"{}\"", state.trs.to_string().lines().join(" "));
                        StateKind::Revision(state)
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
                        state.playout = None;
                        println!("#   \"{}\"", state.trs.to_string().lines().join(" "));
                        StateKind::Revision(state)
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
                StateKind::Revision(state)
            }
            MCTSMove::RegenerateThisRule(n, ref context) => {
                // You're stating where you want to regenerate: state changes.
                let mut state = self.clone();
                state.spec = Some(MCTSMoveState::RegenerateRule(Some((n, context.clone()))));
                StateKind::Revision(state)
            }
        }
    }
    pub fn playout<R: Rng>(&self, mcts: &TRSMCTS<'a, 'b>, rng: &mut R) -> TRS<'a, 'b> {
        let mut trs = self.trs.clone();
        let mut n_revisions = self.n;
        // Finish whatever move is in progress.
        match &self.spec {
            None => (),
            Some(MCTSMoveState::DeleteRules(progress)) => {
                println!("#        finishing deletion");
                let lower_bound = progress.unwrap_or(0);
                for idx in (lower_bound..trs.len()).rev() {
                    if rng.gen() {
                        trs.utrs.remove_idx(idx).ok();
                    }
                }
                n_revisions += 1;
            }
            Some(MCTSMoveState::MemorizeData(progress)) => {
                println!("#         finishing memorization");
                let lower_bound = progress.unwrap_or(0);
                for rule in mcts.data.iter().skip(lower_bound) {
                    if rng.gen() {
                        trs.append_clauses(vec![rule.clone()]).ok();
                    }
                }
                n_revisions += 1;
            }
            Some(MCTSMoveState::RegenerateRule(progress)) => {
                let (n, context) = progress.clone().unwrap_or_else(|| {
                    let idx = (0..trs.len()).choose(rng).unwrap();
                    let rulecontext = RuleContext::from(trs.utrs.rules[idx].clone());
                    let (_, place) = rulecontext.subcontexts().into_iter().choose(rng).unwrap();
                    let context = rulecontext.replace(&place, Context::Hole).unwrap();
                    (idx, context)
                });
                println!(
                    "#         finishing regeneration with: {}",
                    context.pretty(&trs.lex.signature())
                );
                loop {
                    println!("#           looping");
                    if let Ok(rule) = trs
                        .lex
                        .sample_rule_from_context(
                            context.clone(),
                            mcts.atom_weights,
                            mcts.invent,
                            mcts.max_size,
                            rng,
                        )
                        .drop()
                    {
                        println!("#           sampled: {}", rule.pretty(&trs.lex.signature()));
                        trs.append_clauses(vec![rule]).ok();
                        trs.utrs.rules.swap_remove(n);
                        break;
                    }
                }
                n_revisions += 1;
            }
            Some(MCTSMoveState::SampleRule(ref context)) => {
                println!(
                    "#         finalizing sample: {}",
                    context.pretty(&trs.lex.signature())
                );
                loop {
                    println!("#           looping");
                    if let Ok(rule) = trs
                        .lex
                        .sample_rule_from_context(
                            context.clone(),
                            mcts.atom_weights,
                            mcts.invent,
                            mcts.max_size,
                            rng,
                        )
                        .drop()
                    {
                        println!("#           sampled: {}", rule.pretty(&trs.lex.signature()));
                        trs.append_clauses(vec![rule]).ok();
                        break;
                    }
                }
                n_revisions += 1;
            }
        }
        // Run the remaining moves
        while n_revisions < mcts.max_revisions {
            // data, sample, regenerate, delete, stop
            let moves = WeightedIndex::new(&[
                (!mcts.data.is_empty() as usize as f64),
                1.0,
                (!trs.is_empty() as usize as f64),
                ((trs.len() > 1) as usize as f64),
                1.0,
            ])
            .unwrap();
            match moves.sample(rng) {
                0 => {
                    println!("#         adding data");
                    for rule in mcts.data {
                        if rng.gen() {
                            trs.append_clauses(vec![rule.clone()]).ok();
                        }
                    }
                }
                1 => {
                    let schema = TypeSchema::Monotype(trs.lex.fresh_type_variable());
                    println!("#         sampling a rule");
                    loop {
                        println!("#           looping");
                        if let Ok(rule) = trs
                            .lex
                            .sample_rule(
                                &schema,
                                mcts.atom_weights,
                                mcts.invent,
                                mcts.max_size,
                                rng,
                            )
                            .drop()
                        {
                            println!("#           sampled: {}", rule.pretty(&trs.lex.signature()));
                            trs.append_clauses(vec![rule]).ok();
                            break;
                        }
                    }
                }
                2 => {
                    let idx = (0..trs.len()).choose(rng).unwrap();
                    let rulecontext = RuleContext::from(trs.utrs.rules[idx].clone());
                    let (_, place) = rulecontext.subcontexts().into_iter().choose(rng).unwrap();
                    let context = rulecontext.replace(&place, Context::Hole).unwrap();
                    println!(
                        "#         regenerating: {}",
                        context.pretty(&trs.lex.signature())
                    );
                    loop {
                        println!("#           looping");
                        if let Ok(rule) = trs
                            .lex
                            .sample_rule_from_context(
                                context.clone(),
                                mcts.atom_weights,
                                mcts.invent,
                                mcts.max_size,
                                rng,
                            )
                            .drop()
                        {
                            println!("#           sampled: {}", rule.pretty(&trs.lex.signature()));
                            trs.utrs.remove_idx(idx).ok();
                            trs.utrs.insert_idx(idx, rule).ok();
                            break;
                        }
                    }
                }
                3 => {
                    println!("#         deleting rules");
                    for idx in (0..trs.len()).rev() {
                        if rng.gen() {
                            trs.utrs.remove_idx(idx).ok();
                        }
                    }
                }
                4 => {
                    println!("#         stopping");
                    break;
                }
                _ => unreachable!(),
            }
            n_revisions += 1;
        }
        trs
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
            revisions: vec![],
        }
    }
    pub fn add_state(&mut self, state: StateKind<'a, 'b>) -> MCTSState {
        match state {
            StateKind::Terminal(h) => self.add_terminal(h),
            StateKind::Revision(r) => self.add_revision(r),
        }
    }
    pub fn add_revision(&mut self, state: Revise<'a, 'b>) -> MCTSState {
        self.revisions.push(state);
        let handle = StateHandle::Revision(self.revisions.len() - 1);
        MCTSState { handle }
    }
    pub fn add_terminal(&mut self, state: Hypothesis<'a, 'b>) -> MCTSState {
        self.terminals.push(state);
        let handle = StateHandle::Terminal(self.terminals.len() - 1);
        MCTSState { handle }
    }
    pub fn root(&mut self) -> MCTSState {
        let state = Revise {
            trs: TRS::new_unchecked(&self.lexicon, self.deterministic, self.bg, vec![]),
            spec: None,
            n: 0,
            playout: None,
        };
        self.add_revision(state)
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
            StateHandle::Terminal(th) => {
                println!("#       node is terminal");
                mcts.terminals[th].lposterior
            }
            StateHandle::Revision(rh) => match &mcts.revisions[rh].playout {
                Some(th) => {
                    println!("#       found a playout");
                    mcts.terminals[*th].lposterior
                }
                None => {
                    println!("#       playing out");
                    let trs = mcts.revisions[rh].playout(mcts, rng);
                    println!(
                        "#         simulated: \"{}\"",
                        trs.to_string().lines().join(" ")
                    );
                    let h = Hypothesis::new(trs, &mcts.data, 1.0, mcts.params);
                    let score = h.lposterior;
                    let th = mcts.terminals.len();
                    mcts.terminals.push(h);
                    mcts.revisions[rh].playout = Some(th);
                    score
                }
            },
        }
    }
}

// TODO:
// - Restrict search to hypotheses with finite-likelihood.
//   - Remove moves from outgoing if leads to infinitely bad state.
//   - Prevent moves from being added if infinitely bad.
// - Update posterior computation to be more efficient. Let hypothesis store
//   likelihoods so just incrementally update.
// - add revision moves
// - add_moves_for_data
//   - root can memorize again
//   - single-parent nodes can memorizeOne.
use itertools::Itertools;
use mcts::{MoveEvaluator, MoveInfo, NodeHandle, SearchTree, State, StateEvaluator, MCTS};
use polytype::TypeSchema;
use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng,
};
use term_rewriting::{Atom, Context, Rule, RuleContext};
use trs::{Hypothesis, Lexicon, ModelParams, SampleError, TRS};
use utils::logsumexp;

type NonterminalHandle = usize;
type TerminalHandle = usize;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum StateHandle {
    Nonterminal(NonterminalHandle),
    Terminal(TerminalHandle),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MCTSInternalState<'b> {
    lex: Lexicon<'b>,
    rules: Vec<Rule>,
    context: Option<RuleContext>,
    data: bool,
    revisions: usize,
    playout: Option<TerminalHandle>,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct MCTSState {
    handle: StateHandle,
}

pub struct TRSMCTS<'a, 'b> {
    pub lexicon: Lexicon<'b>,
    pub bg: &'a [Rule],
    pub deterministic: bool,
    pub data: &'a [Rule],
    pub nonterminals: Vec<MCTSInternalState<'b>>,
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

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Move {
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
    Compose,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MCTSMove {
    Data { n: usize },
    Empty,
    SampleAtom { atom: Atom },
    // TODO: all the revision moves
}

impl<'a, 'b> State<TRSMCTS<'a, 'b>> for MCTSState {
    type Move = MCTSMove;
    type MoveList = Vec<Self::Move>;
    fn available_moves(&self, mcts: &mut TRSMCTS) -> Self::MoveList {
        match self.handle {
            StateHandle::Terminal(..) => vec![],
            StateHandle::Nonterminal(sh) => {
                let state = &mut mcts.nonterminals[sh];
                let mut moves = vec![];
                if let Some(rc) = &state.context {
                    if rc.is_empty() {
                        if !state.data {
                            for n in 1..mcts.data.len() {
                                moves.push(MCTSMove::Data { n });
                            }
                        }
                        moves.push(MCTSMove::Empty);
                    }
                    if let Some(place) = rc.leftmost_hole() {
                        state
                            .lex
                            .rulecontext_fillers(&rc, &place)
                            .into_iter()
                            .map(|atom| MCTSMove::SampleAtom { atom })
                            .for_each(|mv| moves.push(mv))
                    }
                }
                //else {
                //    // - revisions: options if we've finished the TRS and aren't in the middle of a multi-step move (e.g. regeneration).
                //    // TODO: all the revisions!
                //}
                moves
            }
        }
    }
    fn make_move<R: Rng>(&self, mv: &Self::Move, mcts: &mut TRSMCTS, _rng: &mut R) -> Self {
        match self.handle {
            StateHandle::Terminal(..) => panic!("inconsistent state: no move from terminal"),
            StateHandle::Nonterminal(sh) => {
                let state = &mcts.nonterminals[sh];
                match *mv {
                    MCTSMove::Data { n } => {
                        let mut rules = state.rules.clone();
                        rules.extend_from_slice(&mcts.data[..n]);
                        let state = MCTSInternalState {
                            rules,
                            lex: state.lex.clone(),
                            data: true,
                            context: state.context.clone(),
                            revisions: state.revisions,
                            playout: None,
                        };
                        mcts.nonterminals.push(state);
                        MCTSState {
                            handle: StateHandle::Nonterminal(mcts.nonterminals.len() - 1),
                        }
                    }
                    MCTSMove::Empty => {
                        let trs = TRS::new_unchecked(
                            &state.lex,
                            mcts.deterministic,
                            mcts.bg,
                            state.rules.clone(),
                        );
                        let hypothesis = Hypothesis::new(trs, &mcts.data, 1.0, mcts.params);
                        mcts.terminals.push(hypothesis);
                        MCTSState {
                            handle: StateHandle::Terminal(mcts.terminals.len() - 1),
                        }
                    }
                    MCTSMove::SampleAtom { atom } => {
                        let rc = state.context.as_ref().expect("how did this move happen?");
                        let place = rc.leftmost_hole().expect("How did this move happen?");
                        if let Some(new_context) = rc.replace(&place, Context::from(atom)) {
                            if let Ok(rule) = new_context.to_rule() {
                                let mut state = state.clone();
                                state.rules.push(rule);
                                state.context =
                                    RuleContext::new(Context::Hole, vec![Context::Hole]);
                                state.playout = None;
                                println!(
                                    "#   {}",
                                    state
                                        .rules
                                        .iter()
                                        .map(|r| r.pretty(&state.lex.signature()))
                                        .join("; ")
                                );
                                mcts.nonterminals.push(state);
                                MCTSState {
                                    handle: StateHandle::Nonterminal(mcts.nonterminals.len() - 1),
                                }
                            } else {
                                let mut state = state.clone();
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
                                state.context = Some(new_context);
                                state.playout = None;
                                mcts.nonterminals.push(state);
                                MCTSState {
                                    handle: StateHandle::Nonterminal(mcts.nonterminals.len() - 1),
                                }
                            }
                        } else {
                            panic!("inconsistent state: type inference failed us somewhere");
                        }
                    } // TODO: all the revision moves
                }
            }
        }
    }
    fn add_moves_for_new_data(&self, moves: &[Self::Move], mcts: &mut TRSMCTS) -> Vec<Self::Move> {
        self.available_moves(mcts)
            .into_iter()
            .filter(|m| !moves.contains(&m))
            .collect()
    }
}

impl<'b> MCTSInternalState<'b> {
    pub fn playout<'a, R: Rng>(&self, mcts: &TRSMCTS<'a, 'b>, rng: &mut R) -> TRS<'a, 'b> {
        if let Some(rc) = &self.context {
            let mut lex = self.lex.clone();
            let mut rules = self.rules.clone();
            // Finish sampling rule as needed.
            if !rc.is_empty() {
                println!(
                    "#         finalizing rule context: {}",
                    rc.pretty(&lex.signature())
                );
                loop {
                    println!("#           looping");
                    if let Ok(rule) = lex
                        .sample_rule_from_context(
                            rc.clone(),
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
            // Now, sample stuff as needed.
            let mut data = self.data;
            loop {
                // data, new_rule, empty
                let moves = WeightedIndex::new(&[
                    (!(data || mcts.data.is_empty()) as usize as f64),
                    1.0,
                    1.0,
                ])
                .unwrap();
                match moves.sample(rng) {
                    0 => {
                        println!("#         adding data");
                        rules.extend_from_slice(&mcts.data);
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
        } else {
            panic!("Inconsistent state: playing out a terminal node");
        }
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
            nonterminals: vec![],
        }
    }
    pub fn root(&mut self) -> MCTSState {
        let state = MCTSInternalState {
            lex: self.lexicon.clone(),
            rules: vec![],
            context: RuleContext::new(Context::Hole, vec![Context::Hole]),
            data: false,
            revisions: 0,
            playout: None,
        };
        self.nonterminals.push(state);
        let handle = StateHandle::Nonterminal(0);
        MCTSState { handle }
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
            StateHandle::Nonterminal(sh) => {
                let state = &mcts.nonterminals[sh];
                match state.playout {
                    Some(th) => {
                        println!("#       found a playout");
                        mcts.terminals[th].lposterior
                    }
                    None => {
                        println!("#       playing out");
                        let trs = state.playout(mcts, rng);
                        println!(
                            "#         simulated: \"{}\"",
                            trs.to_string().lines().join(" ")
                        );
                        let h = Hypothesis::new(trs, &mcts.data, 1.0, mcts.params);
                        let score = h.lposterior;
                        let th = mcts.terminals.len();
                        mcts.terminals.push(h);
                        mcts.nonterminals[sh].playout = Some(th);
                        score
                    }
                }
            }
        }
    }
}

impl Move {
    pub fn take<'a, 'b, R: Rng>(
        &self,
        mcts: &TRSMCTS<'a, 'b>,
        rng: &mut R,
        parents: &[&TRS<'a, 'b>],
    ) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        match *self {
            Move::Compose => parents[0].compose(),
            Move::DeleteRule => parents[0].delete_rule(),
            Move::DeleteRules(t) => parents[0].delete_rules(rng, t),
            Move::Generalize => parents[0].generalize(),
            Move::LocalDifference => parents[0].local_difference(rng),
            Move::Memorize(deterministic) => Ok(TRS::memorize(
                &mcts.lexicon,
                deterministic,
                &mcts.bg,
                &mcts.data,
            )),
            Move::MemorizeOne => parents[0].memorize_one(&mcts.data),
            Move::Recurse(n) => parents[0].recurse(n),
            Move::RegenerateRule(aw, mss) => parents[0].regenerate_rule(aw, mss, rng),
            Move::SampleRule(aw, mss) => parents[0].sample_rule(aw, mss, rng),
            Move::Variablize => parents[0].variablize(),
        }
    }
    pub fn deterministic(&self) -> bool {
        match *self {
            Move::Compose
            | Move::DeleteRule
            | Move::Generalize
            | Move::LocalDifference
            | Move::Memorize(_)
            | Move::MemorizeOne
            | Move::Variablize => true,
            Move::DeleteRules(_)
            | Move::Recurse(_)
            | Move::RegenerateRule(..)
            | Move::SampleRule(..) => false,
        }
    }
    pub fn num_parents(&self) -> usize {
        match *self {
            Move::Memorize(_) => 0,
            _ => 1,
        }
    }
    pub fn applies(&self, parents: &[&TRS], data: &[Rule]) -> bool {
        self.num_parents() == parents.len()
            && match *self {
                Move::Compose
                | Move::DeleteRule
                | Move::DeleteRules(_)
                | Move::Generalize
                | Move::LocalDifference
                | Move::Recurse(_)
                | Move::RegenerateRule(..)
                | Move::Variablize => !parents[0].is_empty(),
                Move::MemorizeOne => !data.is_empty(),
                _ => true,
            }
    }
    pub fn data_sensitive(&self) -> bool {
        match *self {
            Move::Memorize(_) | Move::MemorizeOne => true,
            _ => false,
        }
    }
}

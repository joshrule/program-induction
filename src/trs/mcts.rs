// TODO:
// - Restrict search to hypotheses with finite-likelihood.
//   - Remove moves from outgoing if leads to infinitely bad state.
//   - Prevent moves from being added if infinitely bad.
// - Update posterior computation to be more efficient. Let hypothesis store
//   likelihoods so just incrementally update.
// - New moves become available when data is added:
//   - root can memorize again
//   - single-parent nodes can memorizeOne.
use itertools::Itertools;
use mcts::{MoveEvaluator, MoveInfo, NodeHandle, SearchTree, State, StateEvaluator, MCTS};
use rand::Rng;
use term_rewriting::Rule;
use trs::{Hypothesis, Lexicon, ModelParams, SampleError, TRS};
use utils::logsumexp;

type ObjectHandle = usize;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MoveWrapper {
    pub count: usize,
    pub mov: Move,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MoveName {
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
    // NOTE: removed Combine as its not that useful and our only 2-parent move.
    // Combine,
    Compose,
}

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
    // NOTE: removed Combine as its not that useful and our only 2-parent move.
    // Combine(usize),
    Compose,
}

pub struct TRSMCTS<'a, 'b> {
    pub lexicon: Lexicon<'b>,
    pub bg: &'a [Rule],
    pub obs: &'a [Rule],
    pub trss: Vec<Hypothesis<'a, 'b>>,
    pub max_depth: usize,
    pub max_states: usize,
    pub params: ModelParams,
}

impl<'a, 'b> TRSMCTS<'a, 'b> {
    pub fn new(
        lexicon: Lexicon<'b>,
        bg: &'a [Rule],
        obs: &'a [Rule],
        params: ModelParams,
        max_depth: usize,
        max_states: usize,
    ) -> TRSMCTS<'a, 'b> {
        TRSMCTS {
            lexicon,
            bg,
            obs,
            max_depth,
            max_states,
            params,
            trss: vec![],
        }
    }
}

pub struct MCTSMoveEvaluator;

pub struct MCTSStateEvaluator;

#[derive(Debug, Clone, PartialEq)]
pub struct MCTSState {
    parents: Vec<ObjectHandle>,
    moves: Vec<MoveWrapper>,
}

impl<'a, 'b> MoveEvaluator<TRSMCTS<'a, 'b>> for MCTSMoveEvaluator {
    type MoveEvaluation = f64;
    /// This implementation of `choose` is a modified UCT.
    fn choose<'c, MoveIter>(
        &self,
        moves: MoveIter,
        nh: NodeHandle,
        tree: &SearchTree<TRSMCTS<'a, 'b>>,
    ) -> Option<&'c MoveInfo<TRSMCTS<'a, 'b>>>
    where
        MoveIter: Iterator<Item = &'c MoveInfo<TRSMCTS<'a, 'b>>>,
    {
        let moves = moves.collect::<Vec<_>>();
        let node = tree.node(nh);
        println!("      we have {} options", moves.len());
        moves
            .iter()
            .map(|m| {
                println!("      {:?}", m.mov);
                let score = match m.child {
                    Some(ch) => {
                        // Child's Q/N
                        let child = tree.node(ch);
                        println!(
                            "        there's a child, so it's {}'s q/n: {:.3} / {:.3} + sqrt(ln({:.3}) / {:.3})",
                            ch, child.q.exp(), child.n, node.n, child.n
                        );
                        child.q.exp() / child.n + (node.n.ln() / child.n).sqrt()
                    }
                    None if m.mov.count == 0 => {
                        // subtree's Q/N.
                        println!(
                            "        no child & count = 0, so it's {}'s q/n: {:.3} / {:.3} + sqrt(ln({:.3}) / 1.0)",
                            nh, node.q.exp(), node.n, node.n
                        );
                        // m.mov.count + 1 == 1
                        node.q.exp() / node.n + (node.n.ln() / 1.0).sqrt()
                    }
                    None => {
                        // subtree's Q/N
                        let mean = node.q.exp() / node.n;
                        // mean Q/N produced by previous uses of move.
                        let mut deviation = 0.0;
                        for c in 0..m.mov.count {
                            let mut qs = vec![];
                            let mut n = 0.0;
                            for candidate in &moves {
                                if candidate.mov.mov == m.mov.mov && candidate.mov.count == c {
                                    let c_node = tree.node(candidate.child.unwrap());
                                    qs.push(c_node.q);
                                    n += c_node.n;
                                }
                            }
                            // Don't update if the move failed.
                            if n > 0.0 {
                                deviation += logsumexp(&qs).exp() / n;
                            }
                        }
                        // normalizing constant
                        let count = (m.mov.count + 1) as f64;
                        println!(
                            "        no child & count > 0, so it's ({:.3} + {:.3}) / {:.3} + sqrt(ln({:.3}) / {:.3})",
                            mean, deviation, count, node.n, count,
                        );
                        (mean + deviation) / count + (node.n.ln() / count).sqrt()
                    }
                };
                println!("        score: {:.3}", score);
                (m, score)
            })
            .max_by(|x, y| x.1.partial_cmp(&y.1).expect("found NaN"))
            .map(|(m, _)| {
                println!("      we're going with {:?}", m.mov);
                *m
            })
    }
}

impl<'a, 'b> StateEvaluator<TRSMCTS<'a, 'b>> for MCTSStateEvaluator {
    type StateEvaluation = f64;
    fn evaluate(
        &self,
        state: &<TRSMCTS<'a, 'b> as MCTS>::State,
        mcts: &TRSMCTS<'a, 'b>,
    ) -> Self::StateEvaluation {
        let lps = state
            .parents
            .iter()
            .map(|p| mcts.trss[*p].lposterior)
            .collect::<Vec<_>>();
        logsumexp(&lps) / (state.parents.len().max(1) as f64)
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

impl<'a, 'b> State<TRSMCTS<'a, 'b>> for MCTSState {
    type Move = MoveWrapper;
    type MoveList = Vec<Self::Move>;
    fn available_moves(&self, mcts: &TRSMCTS) -> Self::MoveList {
        let parents = self
            .parents
            .iter()
            .map(|&p| &mcts.trss[p].trs)
            .collect::<Vec<_>>();
        self.moves
            .iter()
            .filter(|m| m.mov.applies(&parents, &mcts.obs))
            .copied()
            .collect()
    }
    fn make_move<R: Rng>(&self, mov: &Self::Move, gp: &mut TRSMCTS, rng: &mut R) -> Vec<Self> {
        let parents = self
            .parents
            .iter()
            .map(|&p| &gp.trss[p].trs)
            .collect::<Vec<_>>();
        mov.mov
            .take(gp, rng, &parents)
            .map(|mut trss| {
                let mut unique_trss = Vec::with_capacity(trss.len());
                while let Some(trs) = trss.pop() {
                    if trs.unique_shape(&unique_trss) {
                        unique_trss.push(trs)
                    }
                }
                unique_trss
                    .into_iter()
                    .map(|trs| {
                        println!("{}", trs.to_string().lines().join(" "));
                        let handle: ObjectHandle = gp
                            .trss
                            .iter()
                            .position(|obj| TRS::same_shape(&obj.trs, &trs))
                            .unwrap_or_else(|| gp.trss.len());
                        if handle == gp.trss.len() {
                            println!("  it's a new hypothesis with handle {}", handle);
                            gp.trss.push(Hypothesis::new(trs, &gp.obs, 1.0, gp.params));
                        } else {
                            println!(
                                "  we already have this hypothesis with handle {} {}",
                                handle,
                                TRS::same_shape(&gp.trss[handle].trs, &trs)
                            );
                            println!(
                                "    {}\n    {}",
                                gp.trss[handle].trs.to_string().lines().join(" "),
                                trs.to_string().lines().join(" ")
                            );
                        }
                        MCTSState {
                            // HACK: we need to figure out what sets of parents to create from the moves.
                            parents: vec![handle],
                            moves: self.moves.clone(),
                        }
                    })
                    .collect()
            })
            .unwrap_or_else(|_| vec![])
    }
    fn uniquify(taken: &Self::Move, generated: usize) -> (Vec<Self::Move>, Vec<Self::Move>) {
        println!("    uniquifying {:?}", taken);
        let new_moves = if taken.mov.deterministic() {
            println!("      deterministic, so no new moves");
            vec![]
        } else {
            println!("      nondeterministic, so 1 new move");
            let mut new_move = *taken;
            new_move.count += 1;
            vec![new_move]
        };
        let child_moves = (0..generated).map(|_| *taken).collect();
        (new_moves, child_moves)
    }
    fn add_moves_for_new_data(
        &self,
        moves: &[(&Self::Move, bool)],
        mcts: &TRSMCTS,
    ) -> Vec<Self::Move> {
        self.available_moves(mcts)
            .into_iter()
            .filter(|m| m.mov.data_sensitive())
            .filter_map(|mut m| {
                let recent = moves
                    .iter()
                    .filter(|(mov, _)| mov.mov == m.mov)
                    .max_by_key(|(mov, _)| mov.count);
                if let Some(recent) = recent {
                    if recent.1 {
                        m.count = recent.0.count + 1;
                        Some(m)
                    } else {
                        None
                    }
                } else {
                    Some(m)
                }
            })
            .collect()
    }
}

impl MCTSState {
    pub fn new(parents: Vec<ObjectHandle>, moves: Vec<Move>) -> Self {
        let moves = moves
            .into_iter()
            .map(|m| MoveWrapper { count: 0, mov: m })
            .collect();
        MCTSState { parents, moves }
    }
}

impl Move {
    pub fn take<'a, 'b, R: Rng>(
        &self,
        gp: &TRSMCTS<'a, 'b>,
        rng: &mut R,
        parents: &[&TRS<'a, 'b>],
    ) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        match *self {
            Move::Compose => parents[0].compose(),
            Move::DeleteRule => parents[0].delete_rule(),
            Move::DeleteRules(t) => parents[0].delete_rules(rng, t),
            Move::Generalize => parents[0].generalize(),
            Move::LocalDifference => parents[0].local_difference(rng),
            Move::Memorize(deterministic) => {
                Ok(TRS::memorize(&gp.lexicon, deterministic, &gp.bg, &gp.obs))
            }
            Move::MemorizeOne => parents[0].memorize_one(&gp.obs),
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

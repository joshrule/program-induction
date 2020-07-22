//! Representations capable of [Monte Carlo Tree Search].
//!
//! [Monte Carlo Tree Search]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

use generational_arena::{Arena, Index};
use rand::Rng;
use serde::Serialize;
//use serde_json::Value;
use std::hash::Hash;

pub type Stats<M> = <<M as MCTS>::MoveEval as MoveEvaluator<M>>::NodeStatistics;
type Move<M> = <<M as MCTS>::State as State<M>>::Move;
type Data<M> = <<M as MCTS>::State as State<M>>::Data;
type StateEvaluation<M> = <<M as MCTS>::StateEval as StateEvaluator<M>>::StateEvaluation;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct NodeHandle(Index);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct MoveHandle(Index);

pub trait State<M: MCTS<State = Self>>: Copy + std::hash::Hash + Eq + Sized {
    type Data: std::fmt::Debug + Clone + Sized;
    type Move: std::fmt::Display + PartialEq + Clone;
    type MoveList: IntoIterator<Item = Self::Move>;
    fn root_data(mcts: &M) -> Self::Data;
    fn valid_data(data: &Self::Data, mcts: &M) -> bool;
    fn available_moves(&self, data: &mut Self::Data, depth: usize, mcts: &M) -> Self::MoveList;
    fn make_move(&self, data: &mut Self::Data, mov: &Self::Move, n: usize, mcts: &M);
    fn make_state(data: &Self::Data, mcts: &mut M) -> Option<Self>;
    fn discard(&self, mcts: &mut M);
    //fn describe_self(&self, data: &Self::Data, mcts: &M) -> Value;
    //fn describe_move(&self, data: &Self::Data, mv: &Self::Move, mcts: &M, failed: bool) -> Value;
}

pub trait NodeStatistic<M: MCTS> {
    fn new() -> Self;
    fn update(&mut self, evaluation: StateEvaluation<M>);
    fn combine(&mut self, other: &Self);
}

pub trait MoveEvaluator<M: MCTS<MoveEval = Self>>: Sized {
    type NodeStatistics: std::fmt::Debug + Serialize + NodeStatistic<M> + Clone + Sized;
    fn choose<R: Rng, MoveIter>(
        &self,
        moves: MoveIter,
        node: NodeHandle,
        tree: &SearchTree<M>,
        rng: &mut R,
    ) -> Option<MoveHandle>
    where
        MoveIter: Iterator<Item = MoveHandle>;
}

pub trait StateEvaluator<M: MCTS<StateEval = Self>>: Sized {
    type StateEvaluation: Copy + Into<f64>;
    fn evaluate<R: Rng>(
        &self,
        state: &M::State,
        data: &Data<M>,
        mcts: &mut M,
        rng: &mut R,
    ) -> Self::StateEvaluation;
    fn reread(&self, state: &M::State, mcts: &mut M) -> Self::StateEvaluation;
}

pub trait MCTS: Sized {
    type StateEval: StateEvaluator<Self>;
    type MoveEval: MoveEvaluator<Self>;
    type State: State<Self>;
    fn max_depth(&self) -> usize {
        std::usize::MAX
    }
    fn max_states(&self) -> usize {
        std::usize::MAX
    }
}

pub struct MCTSManager<M: MCTS> {
    tree: SearchTree<M>,
}

pub struct SearchTree<M: MCTS> {
    mcts: M,
    tree: TreeStore<M>,
    state_eval: M::StateEval,
    move_eval: M::MoveEval,
}

pub struct TreeStore<M: MCTS> {
    root: NodeHandle,
    nodes: Arena<Node<M>>,
    moves: Arena<MoveInfo<M>>,
}

pub struct Node<M: MCTS> {
    pub state: M::State,
    pub incoming: Option<MoveHandle>,
    pub outgoing: Vec<MoveHandle>,
    pub evaluation: StateEvaluation<M>,
    pub stats: Stats<M>,
}

pub struct MoveInfo<M: MCTS> {
    pub parent: NodeHandle,
    pub child: Option<NodeHandle>,
    pub mov: Move<M>,
    pub pruning: Pruning,
}

#[derive(Copy, Clone, Debug)]
pub enum MCTSError {
    MoveCreatedCycle,
    MoveFailed,
    TreeExhausted,
    TreeAtMaxStates,
    TreeAtMaxDepth,
    TreeInconsistent,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Pruning {
    None,
    Soft,
    Hard,
}

pub enum MoveCheck<M: MCTS> {
    Failed,
    Expected,
    NewState(M::State),
}

impl std::fmt::Display for MCTSError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            MCTSError::MoveCreatedCycle => write!(f, "move created cycle"),
            MCTSError::MoveFailed => write!(f, "move failed"),
            MCTSError::TreeExhausted => write!(f, "tree exhausted"),
            MCTSError::TreeAtMaxStates => write!(f, "tree contains maximum number of states"),
            MCTSError::TreeAtMaxDepth => write!(f, "tree full to maximum depth"),
            MCTSError::TreeInconsistent => write!(f, "tree is inconsistent"),
        }
    }
}
impl std::error::Error for MCTSError {}

impl<M: MCTS> MCTSManager<M> {
    pub fn new<R: Rng>(
        mcts: M,
        root: M::State,
        state_eval: M::StateEval,
        move_eval: M::MoveEval,
        rng: &mut R,
    ) -> Self {
        let tree = SearchTree::new(mcts, root, state_eval, move_eval, rng);
        MCTSManager { tree }
    }
    // Search until the predicate evaluates to `true`.
    pub fn step_until<R: Rng, P: Fn(&M) -> bool>(&mut self, rng: &mut R, predicate: P) -> usize {
        let mut steps = 0;
        while !predicate(&self.tree.mcts) {
            match self.tree.step(rng) {
                Ok(_nh) => steps += 1,
                Err(e) => match e {
                    MCTSError::TreeInconsistent
                    | MCTSError::TreeExhausted
                    | MCTSError::TreeAtMaxStates
                    | MCTSError::TreeAtMaxDepth => {
                        break;
                    }
                    MCTSError::MoveFailed | MCTSError::MoveCreatedCycle => (),
                },
            }
        }
        steps
    }
    // Take a single search step.
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> Result<NodeHandle, MCTSError> {
        self.tree.step(rng)
    }
    pub fn tree(&self) -> &SearchTree<M> {
        &self.tree
    }
    pub fn tree_mut(&mut self) -> &mut SearchTree<M> {
        &mut self.tree
    }
}

impl<M: MCTS> std::ops::Index<NodeHandle> for Arena<Node<M>> {
    type Output = Node<M>;
    fn index(&self, index: NodeHandle) -> &Self::Output {
        &self[index.0]
    }
}

impl<M: MCTS> std::ops::IndexMut<NodeHandle> for Arena<Node<M>> {
    fn index_mut(&mut self, index: NodeHandle) -> &mut Self::Output {
        &mut self[index.0]
    }
}

impl<M: MCTS> std::ops::Index<MoveHandle> for Arena<MoveInfo<M>> {
    type Output = MoveInfo<M>;
    fn index(&self, index: MoveHandle) -> &Self::Output {
        &self[index.0]
    }
}

impl<M: MCTS> std::ops::IndexMut<MoveHandle> for Arena<MoveInfo<M>> {
    fn index_mut(&mut self, index: MoveHandle) -> &mut Self::Output {
        &mut self[index.0]
    }
}

impl<M: MCTS> TreeStore<M> {
    /// Return a specific node.
    pub fn node(&self, node: NodeHandle) -> &Node<M> {
        &self.nodes[node]
    }
    /// Return a specific move.
    pub fn mv(&self, mh: MoveHandle) -> &MoveInfo<M> {
        &self.moves[mh]
    }
    /// Return the number of DAG nodes.
    pub fn tree_size(&self) -> usize {
        self.nodes.len()
    }
    /// Find the path to this node.
    pub fn path_tree(&self, node: NodeHandle) -> Vec<MoveHandle> {
        let mut path = self.nodes[node].incoming.into_iter().collect::<Vec<_>>();
        while let Some(last) = path.last() {
            let mv = &self.moves[*last];
            match self.nodes[mv.parent].incoming {
                None => {
                    path.reverse();
                    return path;
                }
                Some(mh) => {
                    path.push(mh);
                }
            }
        }
        path
    }
    pub fn depth(&self, mut nh: NodeHandle) -> usize {
        let mut depth = 0;
        while let Some(mh) = self.nodes[nh].incoming {
            depth += 1;
            nh = self.moves[mh].parent;
        }
        depth
    }
    pub fn ancestors_tree(&self, src_nh: NodeHandle) -> Vec<NodeHandle> {
        let mut maybe = Some(src_nh);
        let mut ancestors = vec![src_nh];
        while let Some(nh) = maybe.take() {
            if let Some(mh) = self.nodes[nh].incoming {
                maybe = Some(self.moves[mh].parent);
                ancestors.push(self.moves[mh].parent)
            }
        }
        ancestors
    }
    /// Find all moves sharing this exact parent.
    pub fn siblings_tree(&self, mh: MoveHandle) -> Vec<MoveHandle> {
        self.nodes[self.moves[mh].parent].outgoing.clone()
    }
    fn parent_move(&self, mh: MoveHandle) -> Option<MoveHandle> {
        self.nodes[self.moves[mh].parent].incoming
    }
}

impl<M: MCTS> SearchTree<M> {
    pub fn new<R: Rng>(
        mut mcts: M,
        root_state: M::State,
        state_eval: M::StateEval,
        move_eval: M::MoveEval,
        rng: &mut R,
    ) -> Self {
        let mut nodes = Arena::new();
        let mut moves = Arena::new();

        let mut data = M::State::root_data(&mcts);
        let evaluation = state_eval.evaluate(&root_state, &data, &mut mcts, rng);
        let mut stats = <<M as MCTS>::MoveEval as MoveEvaluator<M>>::NodeStatistics::new();
        stats.update(evaluation);
        let root_node = Node {
            state: root_state,
            incoming: None,
            outgoing: vec![],
            stats,
            evaluation,
        };
        let root = NodeHandle(nodes.insert(root_node));
        let mhs = root_state
            .available_moves(&mut data, 0, &mcts)
            .into_iter()
            .map(|mov| MoveInfo {
                parent: root,
                mov,
                child: None,
                pruning: Pruning::None,
            })
            .map(|mv| MoveHandle(moves.insert(mv)))
            .collect();
        nodes[root].outgoing = mhs;
        let tree = TreeStore { moves, root, nodes };
        SearchTree {
            mcts,
            state_eval,
            move_eval,
            tree,
        }
    }
    fn set_root<R: Rng>(&mut self, root_state: M::State, rng: &mut R) {
        let mut data = M::State::root_data(&self.mcts);
        let evaluation = self
            .state_eval
            .evaluate(&root_state, &data, &mut self.mcts, rng);
        let mut stats = <<M as MCTS>::MoveEval as MoveEvaluator<M>>::NodeStatistics::new();
        stats.update(evaluation);
        let root_node = Node {
            state: root_state,
            incoming: None,
            outgoing: vec![],
            stats,
            evaluation,
        };
        let root = NodeHandle(self.tree.nodes.insert(root_node));
        let mhs = root_state
            .available_moves(&mut data, 0, &self.mcts)
            .into_iter()
            .map(|mov| MoveInfo {
                parent: root,
                mov,
                child: None,
                pruning: Pruning::None,
            })
            .map(|mv| MoveHandle(self.tree.moves.insert(mv)))
            .collect();
        self.tree.nodes[root].outgoing = mhs;
    }
    pub fn reset<R: Rng>(&mut self, root_state: M::State, rng: &mut R) {
        self.tree.nodes.clear();
        self.tree.moves.clear();
        self.set_root(root_state, rng);
    }
    pub fn prune_except_top<I, R: Rng>(&mut self, paths: I, root_state: M::State, rng: &mut R)
    where
        I: Iterator<Item = Vec<Move<M>>>,
    {
        self.reset(root_state, rng);
        for path in paths {
            self.follow_path(&path, rng).ok();
        }
    }
    // TODO: reinstate
    //pub fn to_file(&self, data_file: &str) -> std::io::Result<()> {
    //    let moves = self
    //        .tree
    //        .moves
    //        .iter()
    //        .map(|(mh, mv)| {
    //            json!({
    //                "handle": mh,
    //                "parent": mv.parent,
    //                "child": mv.child,
    //                "move": self.tree.nodes[mv.parent].state.describe_move(&mv.mov, &self.mcts, mv.pruning == Pruning::Hard),
    //                "pruning": mv.pruning,
    //            })
    //        })
    //        .collect::<Vec<_>>();
    //    let nodes = self
    //        .tree
    //        .nodes
    //        .iter()
    //        .map(|(h, n)| {
    //            json!({
    //                "handle": h,
    //                "state": n.state.describe_self(&self.mcts),
    //                "in": n.incoming,
    //                "out": n.outgoing,
    //                "score": n.evaluation.clone().into(),
    //                "stats": n.stats,
    //            })
    //        })
    //        .collect::<Vec<_>>();
    //    let tree = json!({
    //        "root": self.tree.root,
    //        "moves": moves,
    //        "nodes": nodes,
    //    });

    //    let out_file = std::fs::File::create(data_file)?;
    //    serde_json::to_writer(out_file, &tree)?;
    //    Ok(())
    //}
    pub fn mcts(&self) -> &M {
        &self.mcts
    }
    pub fn mcts_mut(&mut self) -> &mut M {
        &mut self.mcts
    }
    pub fn tree(&self) -> &TreeStore<M> {
        &self.tree
    }
    pub fn tree_mut(&mut self) -> &mut TreeStore<M> {
        &mut self.tree
    }
    pub fn reevaluate_states(&mut self) {
        self.reevaluate_nodes();
        self.recompute_qs();
    }
    /// Iterate over the nodes and copy the evaluation information from the objects.
    pub fn reevaluate_nodes(&mut self) {
        for (_, node) in self.tree.nodes.iter_mut() {
            node.evaluation = self.state_eval.reread(&node.state, &mut self.mcts);
        }
    }
    /// Backpropagate over the entire tree.
    pub fn recompute_qs(&mut self) {
        let mut stack = vec![(self.tree.root, false)];
        while let Some((nh, ready)) = stack.pop() {
            if ready {
                let mut stats = <<M as MCTS>::MoveEval as MoveEvaluator<M>>::NodeStatistics::new();
                stats.update(self.tree.nodes[nh].evaluation);
                self.tree.nodes[nh]
                    .outgoing
                    .iter()
                    .copied()
                    .filter(|mh| self.tree.moves[*mh].pruning != Pruning::Hard)
                    .filter_map(|mh| self.tree.moves[mh].child)
                    .for_each(|child| stats.combine(&self.tree.nodes[child].stats));
                self.tree.nodes[nh].stats = stats;
            } else {
                stack.push((nh, true));
                self.tree.nodes[nh]
                    .outgoing
                    .iter()
                    .copied()
                    .filter(|mh| self.tree.moves[*mh].pruning != Pruning::Hard)
                    .filter_map(|mh| self.tree.moves[mh].child)
                    .for_each(|child| stack.push((child, false)));
            }
        }
    }
    pub fn follow_path<R: Rng>(&mut self, path: &[Move<M>], rng: &mut R) -> Result<(), MCTSError> {
        let mut nh = self.tree.root;
        let mut data = M::State::root_data(&self.mcts);
        for mv in path {
            self.can_continue()?;
            let mh = *self.tree.nodes[nh]
                .outgoing
                .iter()
                .find(|mh| self.tree.moves[**mh].mov == *mv)
                .ok_or(MCTSError::MoveFailed)?;
            match self.expand_node(nh, mh, &mut data)? {
                None => nh = self.tree.moves[mh].child.expect("inconsistent tree"),
                Some(child_state) => {
                    let ph = self.tree.moves[mh].parent;
                    let depth = self.tree.depth(ph);
                    if let Ok(ch) =
                        self.update_tree(Some(child_state), &mut data.clone(), ph, &mv, depth, rng)
                    {
                        nh = ch;
                        self.backpropagate_tree(ch);
                        self.soft_prune_tree(mh);
                    }
                }
            }
        }
        Ok(())
    }
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> Result<NodeHandle, MCTSError> {
        // Check that you can continue.
        self.can_continue()?;
        // Iterate down through the tree.
        let mut nh = self.tree.root;
        let mut data = M::State::root_data(&self.mcts);
        for _ in 0..=self.mcts.max_depth() {
            // Choose a move. Avoid moves going to pruned nodes.
            let moves = self.tree.nodes[nh]
                .outgoing
                .iter()
                .copied()
                .filter(|mh| self.tree.moves[*mh].pruning == Pruning::None);
            let n = self.tree.nodes[nh]
                .outgoing
                .iter()
                .filter(|mh| self.tree.moves[**mh].pruning == Pruning::None)
                .count();
            let mh = self
                .move_eval
                .choose(moves, nh, &self, rng)
                .expect("INVARIANT: active nodes must have moves");
            let mov = &self.tree.moves[mh];
            self.tree.nodes[nh]
                .state
                .make_move(&mut data, &mov.mov, n, &self.mcts);
            match mov.child {
                // Descend known moves.
                Some(child_nh) => {
                    nh = child_nh;
                }
                // Take new moves.
                None => {
                    // Make the child state.
                    let child_state = M::State::make_state(&data, &mut self.mcts);
                    // Get the parent entry.
                    let ph = self.tree.moves[mh].parent;
                    // For each parent, update the tree, backpropagate, etc.
                    let mov = &self.tree.moves[mh].mov.clone();
                    let depth = self.tree.depth(ph);
                    let ch = self.update_tree(child_state, &mut data, ph, &mov, depth, rng)?;
                    self.backpropagate_tree(ch);
                    self.soft_prune_tree(mh);
                    return Ok(ch);
                }
            }
        }
        Err(MCTSError::TreeAtMaxDepth)
    }
    fn can_continue(&self) -> Result<(), MCTSError> {
        if self.tree.nodes.len() >= self.mcts.max_states() {
            Err(MCTSError::TreeAtMaxStates)
        } else if !self.tree.nodes[self.tree.root]
            .outgoing
            .iter()
            .any(|mh| self.tree.moves[*mh].pruning == Pruning::None)
        {
            Err(MCTSError::TreeExhausted)
        } else {
            Ok(())
        }
    }
    // Expand the search tree by taking a single move.
    fn expand_node(
        &mut self,
        nh: NodeHandle,
        mh: MoveHandle,
        data: &mut Data<M>,
    ) -> Result<Option<M::State>, MCTSError> {
        let n = self.tree.nodes[nh].outgoing.len();
        let mv = &self.tree.moves[mh];
        self.tree.nodes[nh]
            .state
            .make_move(data, &mv.mov, n, &self.mcts);
        match mv.child {
            Some(_) => Ok(None),
            None => Ok(M::State::make_state(&data, &mut self.mcts)),
        }
    }
    fn update_tree<R: Rng>(
        &mut self,
        move_result: Option<M::State>,
        data: &mut Data<M>,
        nh: NodeHandle,
        mov: &Move<M>,
        depth: usize,
        rng: &mut R,
    ) -> Result<NodeHandle, MCTSError> {
        // Find the move handle.
        let mh = self.tree.nodes[nh]
            .outgoing
            .iter()
            .copied()
            .find(|mh| self.tree.moves[*mh].mov == *mov)
            .ok_or(MCTSError::TreeInconsistent)?;
        match move_result {
            None => {
                let pmh = self.tree.parent_move(mh);
                self.hard_prune_tree(mh);
                if let Some(pmh) = pmh {
                    self.soft_prune_tree(pmh);
                }
                Err(MCTSError::MoveFailed)
            }
            Some(child_state) => {
                let ancestors = self.tree.ancestors_tree(nh);
                let ch = self.make_node(child_state, data, depth, rng);
                // Prevent cycles: don't add nodes whose state is contained in an ancestor.
                if ancestors.iter().any(|a| ch == *a) {
                    let pmh = self.tree.parent_move(mh);
                    self.hard_prune_tree(mh);
                    if let Some(pmh) = pmh {
                        self.soft_prune_tree(pmh);
                    }
                    Err(MCTSError::MoveCreatedCycle)
                } else {
                    // Give the move a child.
                    self.tree.moves[mh].child = Some(ch);
                    // Give the child an incoming edge.
                    self.tree.nodes[ch].incoming.replace(mh);
                    Ok(ch)
                }
            }
        }
    }
    fn make_node<R: Rng>(
        &mut self,
        state: M::State,
        data: &mut Data<M>,
        depth: usize,
        rng: &mut R,
    ) -> NodeHandle {
        let evaluation = self.state_eval.evaluate(&state, data, &mut self.mcts, rng);
        let moves = state.available_moves(data, depth, &self.mcts);
        let node = Node {
            state,
            incoming: None,
            outgoing: vec![],
            evaluation,
            stats: <<M as MCTS>::MoveEval as MoveEvaluator<M>>::NodeStatistics::new(),
        };
        let nh = NodeHandle(self.tree.nodes.insert(node));
        for mov in moves {
            let new_move = MoveInfo {
                mov,
                parent: nh,
                child: None,
                pruning: Pruning::None,
            };
            let mh = MoveHandle(self.tree.moves.insert(new_move));
            self.tree.nodes[nh].outgoing.push(mh);
        }
        nh
    }
    /// Prunes paths descending from failed/cyclic moves.
    fn hard_prune_tree(&mut self, src_mh: MoveHandle) {
        let mut stack = vec![src_mh];
        while let Some(mh) = stack.pop() {
            if let Some(nh) = self.tree.moves[mh].child {
                for &omh in &self.tree.nodes[nh].outgoing {
                    stack.push(omh);
                }
                self.tree.nodes[nh].state.discard(&mut self.mcts);
                self.tree.nodes.remove(nh.0);
            }
            // Remove all but the src_mh, which is marked pruned.
            if mh != src_mh {
                self.tree.moves.remove(mh.0);
            } else {
                self.tree.moves[mh].pruning = Pruning::Hard;
                self.tree.moves[mh].child = None;
            }
        }
    }
    /// Prunes paths that cannot lead to new `Node`s.
    fn soft_prune_tree(&mut self, src_mh: MoveHandle) {
        let mut maybe = Some(src_mh);
        while let Some(mh) = maybe.take() {
            let mv = &self.tree.moves[mh];
            if mv.pruning != Pruning::Hard {
                if let Some(nh) = mv.child {
                    let outgoing = &self.tree.nodes[nh].outgoing;
                    if outgoing
                        .iter()
                        .all(|mh| self.tree.moves[*mh].pruning != Pruning::None)
                    {
                        self.tree.moves[mh].pruning = Pruning::Soft;
                        maybe = self.tree.parent_move(mh);
                    }
                }
            }
        }
    }
    // Bubble search statistics and pruning information back to the root.
    fn backpropagate_tree(&mut self, new_node: NodeHandle) {
        let mut maybe = Some(new_node);
        while let Some(nh) = maybe.take() {
            // Update stats.
            let evaluation = self.tree.nodes[new_node].evaluation;
            self.tree.nodes[nh].stats.update(evaluation);
            // Add nodes to stack as appropriate.
            maybe = self.tree.nodes[nh]
                .incoming
                .map(|mh| self.tree.moves[mh].parent);
        }
    }
}

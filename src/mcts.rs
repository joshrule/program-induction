//! Representations capable of [Monte Carlo Tree Search].
//!
//! [Monte Carlo Tree Search]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

// TODO:
// - ask for states to be Debug for easier debugging.
use generational_arena::{Arena, Index};
use itertools::Itertools;
use rand::Rng;
use serde::Serialize;
//use serde_json::Value;
use std::{collections::HashMap, hash::Hash};

pub type Stats<M> = <<M as MCTS>::MoveEval as MoveEvaluator<M>>::NodeStatistics;
type Move<M> = <<M as MCTS>::State as State<M>>::Move;
type Data<M> = <<M as MCTS>::State as State<M>>::Data;
type Key<M> = <<M as MCTS>::State as State<M>>::Key;
type StateEvaluation<M> = <<M as MCTS>::StateEval as StateEvaluator<M>>::StateEvaluation;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct NodeHandle(Index);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct MoveHandle(Index);

pub trait State<M: MCTS<State = Self>>: Copy + std::hash::Hash + Eq + Sized {
    type Data: std::fmt::Debug + Clone + Sized;
    type Key: std::fmt::Debug + Copy + Eq + Hash + Sized;
    type Move: std::fmt::Display + PartialEq + Clone;
    type MoveList: IntoIterator<Item = Self::Move>;
    fn root_data(mcts: &M) -> Self::Data;
    fn valid_data(data: &Self::Data, mcts: &M) -> bool;
    fn available_moves(&self, data: &mut Self::Data, depth: usize, mcts: &M) -> Self::MoveList;
    fn make_move(&self, data: &mut Self::Data, mov: &Self::Move, n: usize, mcts: &M);
    fn make_state(data: &Self::Data, mcts: &mut M) -> Option<Self>;
    fn key(&self) -> Self::Key;
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
    table: HashMap<Key<M>, Vec<NodeHandle>>,
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
                        println!("Stopping trial because: {}", e);
                        break;
                    }
                    MCTSError::MoveFailed | MCTSError::MoveCreatedCycle => (),
                },
            }
        }
        steps
    }
    // Take a single search step.
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> Result<Vec<NodeHandle>, MCTSError> {
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
    pub fn dag_size(&self) -> usize {
        self.table.len()
    }
    /// Return the number of DAG nodes.
    pub fn tree_size(&self) -> usize {
        self.nodes.len()
    }
    /// Find the paths to equivalent nodes.
    pub fn paths_dag(&self, nh: NodeHandle) -> Vec<Vec<MoveHandle>> {
        self.table[&self.nodes[nh].state.key()]
            .iter()
            .map(|nh| self.path_tree(*nh))
            .collect()
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
    /// Find all nodes giving rise to equivalent nodes.
    pub fn ancestors_dag(&self, src_nh: NodeHandle) -> Vec<NodeHandle> {
        let mut stack = self.table[&self.nodes[src_nh].state.key()].clone();
        let mut ancestors = vec![];
        while let Some(nh) = stack.pop() {
            if let Some(mh) = self.nodes[nh].incoming {
                let parent = self.moves[mh].parent;
                if !ancestors.contains(&parent) {
                    ancestors.push(parent);
                    stack.push(parent);
                }
            }
        }
        ancestors
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
    /// Find all nodes descending from equivalent nodes.
    pub fn descendants_dag(&self, src_nh: NodeHandle) -> Vec<NodeHandle> {
        let mut stack = self.table[&self.nodes[src_nh].state.key()].clone();
        let mut descendants = vec![];
        while let Some(nh) = stack.pop() {
            for &mh in &self.nodes[nh].outgoing {
                if let Some(ch) = self.moves[mh].child {
                    if !descendants.contains(&ch) {
                        descendants.push(ch);
                        stack.push(ch);
                    }
                }
            }
        }
        descendants
    }
    /// Find all moves sharing an equivalent parent.
    pub fn siblings_dag(&self, src_mh: MoveHandle) -> Vec<MoveHandle> {
        self.table[&self.nodes[self.moves[src_mh].parent].state.key()]
            .iter()
            .flat_map(|nh| self.nodes[*nh].outgoing.iter().copied())
            .collect()
    }
    /// Find all moves sharing this exact parent.
    pub fn siblings_tree(&self, mh: MoveHandle) -> Vec<MoveHandle> {
        self.nodes[self.moves[mh].parent].outgoing.clone()
    }
    /// Remove the first occurrence of some node
    fn delist(&mut self, nh: NodeHandle) {
        if let Some(nodes) = self.table.get_mut(&self.nodes[nh].state.key()) {
            // TODO: convert to if? Entries should be unique.
            while let Some(idx) = nodes.iter().position(|x| *x == nh) {
                nodes.swap_remove(idx);
            }
        }
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
        let mut table = HashMap::new();
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
        table.insert(root_state.key(), vec![root]);
        let tree = TreeStore {
            moves,
            table,
            root,
            nodes,
        };
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
        self.tree.table.insert(root_state.key(), vec![root]);
    }
    pub fn reset<R: Rng>(&mut self, root_state: M::State, rng: &mut R) {
        self.tree.table.clear();
        self.tree.nodes.clear();
        self.tree.moves.clear();
        self.set_root(root_state, rng);
    }
    pub fn prune_except_top<R: Rng>(&mut self, n: usize, root_state: M::State, rng: &mut R) {
        let mut paths = Vec::with_capacity(n);
        for (idx, _) in self.tree.nodes.iter().sorted_by(|a, b| {
            a.1.evaluation
                .into()
                .partial_cmp(&b.1.evaluation.into())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            // Find the best valid path not contained in the set.
            if NodeHandle(idx) != self.tree.root
                && paths
                    .iter()
                    .flatten()
                    .all(|x: &MoveHandle| self.tree.moves[*x].child != Some(NodeHandle(idx)))
            {
                let path = self.tree.path_tree(NodeHandle(idx));
                if self.valid_path(&path) {
                    paths.push(path);
                    // Stop when you have n paths.
                    if paths.len() == n {
                        break;
                    }
                }
            }
        }
        // Convert the paths to moves.
        let moves = paths
            .into_iter()
            .map(|path| {
                path.into_iter()
                    .map(|mh| self.tree.moves[mh].mov.clone())
                    .collect_vec()
            })
            .collect_vec();
        // Reset the tree.
        self.reset(root_state, rng);
        // Take the series of moves specified by each path.
        for path in &moves {
            self.follow_path(path, rng).ok();
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
    //pub fn check_tree<R: Rng>(&mut self, rng: &mut R) {
    //    // 1. Create a new table.
    //    self.tree.table = HashMap::new();
    //    // 2. Iterate over all moves and reset pruning. We do this because some moves may depend on global state.
    //    for (_, mv) in self.tree.moves.iter_mut() {
    //        mv.pruning = Pruning::None;
    //    }
    //    // 3. Process the root (assuming root state is correct).
    //    self.update_moves(self.tree.root);
    //    let root = &mut self.tree.nodes[self.tree.root];
    //    let rh = self.tree.root;
    //    root.incoming = None;
    //    self.tree
    //        .table
    //        .entry(root.state.clone())
    //        .or_insert_with(|| vec![rh]);
    //    // 4. Initialize a stack of MoveInfos to contain the moves leaving the root.
    //    let mut stack = root.outgoing.to_vec();
    //    // 5. Iteratively process the stack:
    //    while let Some(mh) = stack.pop() {
    //        if let Some(ch) = self.tree.moves[mh].child {
    //            // Check whether the move generates the child.
    //            match <M>::State::check_move(mh, &mut self.mcts, &self.tree) {
    //                // If the move fails, hard prune the node.
    //                MoveCheck::Failed => {
    //                    self.hard_prune_tree(mh);
    //                    if let Some(parent_mh) = self.tree.parent_move(mh) {
    //                        self.soft_prune_tree(parent_mh);
    //                    }
    //                }
    //                // If so, update the child's incoming and outgoing, and add it to the table.
    //                MoveCheck::Expected => {
    //                    self.tree.nodes[ch].incoming.replace(mh);
    //                    self.update_moves(ch);
    //                    let entry = self
    //                        .tree
    //                        .table
    //                        .entry(self.tree.nodes[ch].state.clone())
    //                        .or_insert_with(Vec::new);
    //                    entry.push(ch);
    //                }
    //                // If the move gives a new state:
    //                MoveCheck::NewState(new_state) => {
    //                    // Get the ancestors of this particular node.
    //                    let ancestors = self.tree.ancestors_tree(self.tree.moves[mh].parent);
    //                    // Evaluate the node.
    //                    self.tree.nodes[ch].evaluation =
    //                        self.state_eval.evaluate(&new_state, &mut self.mcts, rng);
    //                    // Update the tree's state.
    //                    self.tree.nodes[ch].state = new_state.clone();
    //                    // Hard prune if adding the child would create a cycle, else update.
    //                    let entry = self.tree.table.entry(new_state).or_insert_with(Vec::new);
    //                    if ancestors.iter().any(|a| entry.contains(a)) {
    //                        self.hard_prune_tree(mh);
    //                        if let Some(parent_mh) = self.tree.parent_move(mh) {
    //                            self.soft_prune_tree(parent_mh);
    //                        }
    //                    } else {
    //                        self.tree.nodes[ch].incoming.replace(mh);
    //                        self.update_moves(ch);
    //                        let entry = self
    //                            .tree
    //                            .table
    //                            .entry(self.tree.nodes[ch].state.clone())
    //                            .or_insert_with(Vec::new);
    //                        entry.push(ch);
    //                    }
    //                }
    //            };
    //            // Soft prune or extend stack.
    //            if let Some(node) = self.tree.nodes.get(ch.0) {
    //                if node.outgoing.is_empty() {
    //                    self.soft_prune_tree(mh);
    //                } else {
    //                    stack.extend_from_slice(&node.outgoing)
    //                }
    //            }
    //        }
    //    }
    //}
    ///// Add moves as appropriate.
    //fn update_moves(&mut self, nh: NodeHandle) {
    //    let old_moves = self.tree.nodes[nh]
    //        .outgoing
    //        .iter()
    //        .map(|m| &self.tree.moves[*m].mov)
    //        .collect::<Vec<&Move<M>>>();
    //    let current_moves = self.tree.nodes[nh].state.available_moves(&self.mcts);
    //    let (current_moves, new_moves): (Vec<Move<M>>, Vec<Move<M>>) = current_moves
    //        .into_iter()
    //        .partition(|m| old_moves.contains(&&m));
    //    let (_, failed_moves): (Vec<&Move<M>>, Vec<&Move<M>>) = old_moves
    //        .into_iter()
    //        .partition(|m| current_moves.contains(*m));
    //    let mut pruned = Vec::with_capacity(self.tree.nodes[nh].outgoing.len());
    //    for &mh in &self.tree.nodes[nh].outgoing {
    //        if failed_moves.contains(&&self.tree.moves[mh].mov) {
    //            pruned.push(mh);
    //        }
    //    }
    //    for mh in pruned {
    //        self.hard_prune_tree(mh);
    //    }
    //    for mov in new_moves {
    //        let new_move: MoveInfo<M> = MoveInfo {
    //            parent: nh,
    //            child: None,
    //            mov,
    //            pruning: Pruning::None,
    //        };
    //        let mh = MoveHandle(self.tree.moves.insert(new_move));
    //        self.tree.nodes[nh].outgoing.push(mh);
    //    }
    //}
    /// Take a single search step.
    fn valid_path(&self, path: &[MoveHandle]) -> bool {
        let mut nh = self.tree.root;
        let mut data = M::State::root_data(&self.mcts);
        for mh in path {
            let mv = &self.tree.moves[*mh];
            let n = self.tree.nodes[mv.parent].outgoing.len();
            self.tree.nodes[nh]
                .state
                .make_move(&mut data, &mv.mov, n, &self.mcts);
            match mv.child {
                // Descend known moves.
                Some(child_nh) => {
                    if M::State::valid_data(&data, &self.mcts) {
                        nh = child_nh;
                    } else {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
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
                    let parent_state = &self.tree.nodes[self.tree.moves[mh].parent].state;
                    let parents = self.tree.table[&parent_state.key()].clone();
                    for ph in parents {
                        let depth = self.tree.depth(ph);
                        if let Ok(ch) = self.update_tree(
                            Some(child_state),
                            &mut data.clone(),
                            ph,
                            &mv,
                            depth,
                            rng,
                        ) {
                            let table_mh = self.tree.nodes[ch]
                                .incoming
                                .expect("INVARIANT: child node has parent");
                            nh = ch;
                            self.backpropagate_tree(ch);
                            self.soft_prune_tree(table_mh);
                        }
                    }
                }
            }
        }
        Ok(())
    }
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> Result<Vec<NodeHandle>, MCTSError> {
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
                    let parent_state = &self.tree.nodes[self.tree.moves[mh].parent].state;
                    let parents = self.tree.table[&parent_state.key()].clone();
                    // For each parent, update the tree, backpropagate, etc.
                    let mov = &self.tree.moves[mh].mov.clone();
                    return parents
                        .iter()
                        .filter_map(|ph| {
                            let depth = self.tree.depth(*ph);
                            match self.update_tree(
                                child_state,
                                &mut data.clone(),
                                *ph,
                                &mov,
                                depth,
                                rng,
                            ) {
                                Ok(ch) => {
                                    let table_mh = self.tree.nodes[ch]
                                        .incoming
                                        .expect("INVARIANT: child node has parent");
                                    self.backpropagate_tree(ch);
                                    self.soft_prune_tree(table_mh);
                                    Some(Ok(ch))
                                }
                                Err(_) => None,
                            }
                        })
                        .collect();
                }
            }
        }
        Err(MCTSError::TreeAtMaxDepth)
    }
    fn can_continue(&self) -> Result<(), MCTSError> {
        if self.tree.table.len() >= self.mcts.max_states() {
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
                let entry = self
                    .tree
                    .table
                    .entry(child_state.key())
                    .or_insert_with(Vec::new);
                // Prevent cycles: don't add nodes whose state is contained in an ancestor.
                if ancestors.iter().any(|a| entry.contains(a)) {
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
                    // Add child to DAG.
                    entry.push(ch);
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
    /// Prunes a `MoveInfo<M>` from the tree.
    fn prune_dag(&mut self, src_mh: MoveHandle, prune: Pruning) {
        let src_mv = &self.tree.moves[src_mh];
        let mov = src_mv.mov.clone();
        let parents = self.tree.table[&self.tree.nodes[src_mv.parent].state.key()].clone();
        for &nh in &parents {
            // find the move
            if let Some(mh) = self.tree.nodes[nh]
                .outgoing
                .iter()
                .copied()
                .find(|mh| self.tree.moves[*mh].mov == mov)
            {
                match prune {
                    Pruning::None => (),
                    Pruning::Soft => self.soft_prune_tree(mh),
                    Pruning::Hard => {
                        self.hard_prune_tree(mh);
                        if let Some(parent_mh) = self.tree.parent_move(mh) {
                            self.soft_prune_tree(parent_mh);
                        }
                    }
                }
            }
        }
    }
    /// Prunes paths descending from failed/cyclic moves.
    fn hard_prune_tree(&mut self, src_mh: MoveHandle) {
        let mut stack = vec![src_mh];
        while let Some(mh) = stack.pop() {
            if let Some(nh) = self.tree.moves[mh].child {
                self.tree.delist(nh);
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

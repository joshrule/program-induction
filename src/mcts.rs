//! Representations capable of [Monte Carlo Tree Search].
//!
//! [Monte Carlo Tree Search]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

// TODO:
// - Find States via Hash rather than Eq.
// - ask for moves & states to be Debug for easier debugging.
// - Create a recycling pool for moves & nodes.
use rand::Rng;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

pub type Stats<M> = <<M as MCTS>::MoveEval as MoveEvaluator<M>>::NodeStatistics;
type Move<M> = <<M as MCTS>::State as State<M>>::Move;
type StateEvaluation<M> = <<M as MCTS>::StateEval as StateEvaluator<M>>::StateEvaluation;
pub type NodeHandle = usize;
pub type MoveHandle = usize;

pub trait State<M: MCTS<State = Self>>: Clone + std::hash::Hash + Eq + Sized {
    type Move: std::fmt::Display + PartialEq + Clone;
    type MoveList: IntoIterator<Item = Self::Move>;
    fn available_moves(&self, mcts: &M) -> Self::MoveList;
    fn make_move(
        &self,
        parent: NodeHandle,
        mov: &Self::Move,
        mcts: &mut M,
        tree: &TreeStore<M>,
    ) -> Option<Self>;
    fn add_moves_for_new_data(&self, moves: &[Self::Move], mcts: &mut M) -> Self::MoveList;
    fn check_move(mh: MoveHandle, mcts: &mut M, tree: &TreeStore<M>) -> MoveCheck<M>;
    fn describe_self(&self, mcts: &M) -> Value;
    fn describe_move(&self, mv: &Self::Move, mcts: &M, failed: bool) -> Value;
}

pub trait NodeStatistic<M: MCTS> {
    fn new() -> Self;
    fn update(&mut self, evaluation: StateEvaluation<M>);
    fn combine(&mut self, other: &Self);
}

pub trait MoveEvaluator<M: MCTS<MoveEval = Self>>: Sized {
    type NodeStatistics: std::fmt::Debug + Serialize + NodeStatistic<M> + Clone + Sized;
    fn choose<'a, R: Rng, MoveIter>(
        &self,
        moves: MoveIter,
        node: NodeHandle,
        tree: &SearchTree<M>,
        rng: &mut R,
    ) -> Option<&'a MoveInfo<M>>
    where
        MoveIter: Iterator<Item = &'a MoveInfo<M>>;
}

pub trait StateEvaluator<M: MCTS<StateEval = Self>>: Sized {
    type StateEvaluation: Copy + Into<f64>;
    fn evaluate<R: Rng>(
        &self,
        state: &M::State,
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
    nodes: Vec<Node<M>>,
    moves: Vec<MoveInfo<M>>,
    table: HashMap<M::State, Vec<NodeHandle>>,
}

pub struct Node<M: MCTS> {
    pub state: M::State,
    pub incoming: Option<MoveHandle>,
    pub outgoing: Vec<MoveHandle>,
    pub evaluation: StateEvaluation<M>,
    pub stats: Stats<M>,
}

pub struct MoveInfo<M: MCTS> {
    handle: MoveHandle,
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
            MCTSError::TreeInconsistent => write!(f, "Congratulations! You've exposed a bug"),
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
    pub fn step_until<R: Rng, P: Fn(&M) -> bool>(&mut self, rng: &mut R, predicate: P) {
        while !predicate(&self.tree.mcts) {
            match self.tree.step(rng) {
                Ok(_nh) => {
                }
                Err(e) => {
                    match e {
                        MCTSError::TreeInconsistent
                        | MCTSError::TreeExhausted
                        | MCTSError::TreeAtMaxStates
                        | MCTSError::TreeAtMaxDepth => break,
                        MCTSError::MoveFailed | MCTSError::MoveCreatedCycle => (),
                    }
                }
            }
        }
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

impl<M: MCTS> TreeStore<M> {
    /// Return a specific node.
    pub fn node(&self, node: NodeHandle) -> &Node<M> {
        &self.nodes[node]
    }
    /// Return a specific move.
    pub fn mv(&self, mh: MoveHandle) -> &MoveInfo<M> {
        &self.moves[mh]
    }
    /// Find the paths to equivalent nodes.
    pub fn paths_dag(&self, nh: NodeHandle) -> Vec<Vec<MoveHandle>> {
        self.table[&self.nodes[nh].state]
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
    /// Find all nodes giving rise to equivalent nodes.
    pub fn ancestors_dag(&self, src_nh: NodeHandle) -> Vec<NodeHandle> {
        let mut stack = self.table[&self.nodes[src_nh].state].clone();
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
        let mut stack = self.table[&self.nodes[src_nh].state].clone();
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
        self.table[&self.nodes[self.moves[src_mh].parent].state]
            .iter()
            .flat_map(|nh| self.nodes[*nh].outgoing.iter().copied())
            .collect()
    }
    /// Find all moves sharing this exact parent.
    pub fn siblings_tree(&self, mh: MoveHandle) -> Vec<MoveHandle> {
        self.nodes[self.moves[mh].parent].outgoing.clone()
    }
    fn delist(&mut self, nh: NodeHandle) {
        if let Some(nodes) = self.table.get_mut(&self.nodes[nh].state) {
            if let Some(idx) = nodes.iter().position(|x| *x == nh) {
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
        root: M::State,
        state_eval: M::StateEval,
        move_eval: M::MoveEval,
        rng: &mut R,
    ) -> Self {
        let evaluation = state_eval.evaluate(&root, &mut mcts, rng);
        let moves: Vec<_> = root
            .available_moves(&mut mcts)
            .into_iter()
            .enumerate()
            .map(|(i, mov)| MoveInfo {
                handle: i,
                parent: 0,
                mov,
                child: None,
                pruning: Pruning::None,
            })
            .collect();
        let mut stats = <<M as MCTS>::MoveEval as MoveEvaluator<M>>::NodeStatistics::new();
        stats.update(evaluation);
        let root_node = Node {
            state: root.clone(),
            incoming: None,
            outgoing: (0..moves.len()).collect(),
            stats,
            evaluation,
        };
        let mut table = HashMap::new();
        table.insert(root, vec![0]);
        let tree = TreeStore {
            moves,
            table,
            root: 0,
            nodes: vec![root_node],
        };
        SearchTree {
            mcts,
            state_eval,
            move_eval,
            tree,
        }
    }
    pub fn to_file(&self, data_file: &str) -> std::io::Result<()> {
        let moves = self
            .tree
            .moves
            .iter()
            .enumerate()
            .map(|(i, mv)| {
                json!({
                    "handle": i,
                    "parent": mv.parent,
                    "child": mv.child,
                    "move": self.tree.nodes[mv.parent].state.describe_move(&mv.mov, &self.mcts, mv.pruning == Pruning::Hard),
                    "pruning": mv.pruning,
                })
            })
            .collect::<Vec<_>>();
        let nodes = self
            .tree
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                json!({
                    "handle": i,
                    "state": n.state.describe_self(&self.mcts),
                    "in": n.incoming,
                    "out": n.outgoing,
                    "score": n.evaluation.clone().into(),
                    "stats": n.stats,
                })
            })
            .collect::<Vec<_>>();
        let tree = json!({
            "root": self.tree.root,
            "moves": moves,
            "nodes": nodes,
        });

        let out_file = std::fs::File::create(data_file)?;
        serde_json::to_writer(out_file, &tree)?;
        Ok(())
    }
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
        for node in self.tree.nodes.iter_mut() {
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
    pub fn check_tree<R: Rng>(&mut self, rng: &mut R) {
        // 1. Create a new table.
        self.tree.table = HashMap::new();
        // 2. Process the root (assuming root state is correct).
        self.update_moves(self.tree.root);
        let root = &mut self.tree.nodes[self.tree.root];
        root.incoming = None;
        let entry = self
            .tree
            .table
            .entry(root.state.clone())
            .or_insert_with(|| vec![]);
        entry.push(self.tree.root);
        // 3. Initialize a stack of MoveInfos to contain the moves leaving the root.
        let mut stack = root.outgoing.to_vec();
        // 4. Iteratively process the stack:
        while let Some(mh) = stack.pop() {
            match self.tree.moves[mh].child {
                None => self.tree.moves[mh].pruning = Pruning::None,
                Some(ch) => {
                    match <M>::State::check_move(mh, &mut self.mcts, &self.tree) {
                        MoveCheck::Failed => self.hard_prune_tree(mh),
                        MoveCheck::Expected => (),
                        MoveCheck::NewState(new_state) => {
                            let ancestors = self.tree.ancestors_tree(self.tree.moves[mh].parent);
                            self.tree.nodes[ch].evaluation =
                                self.state_eval.evaluate(&new_state, &mut self.mcts, rng);
                            self.tree.nodes[ch].state = new_state.clone();
                            let entry = self.tree.table.entry(new_state).or_insert_with(|| vec![]);
                            // Prevent cycles: don't add nodes whose state is contained in an ancestor.
                            if ancestors.iter().any(|a| entry.contains(a)) {
                                self.hard_prune_tree(mh);
                            } else {
                                entry.push(ch);
                            }
                        }
                    };
                    self.tree.nodes[ch].incoming.replace(mh);
                    self.update_moves(ch);
                    let entry = self
                        .tree
                        .table
                        .entry(self.tree.nodes[ch].state.clone())
                        .or_insert_with(|| vec![]);
                    entry.push(ch);
                    if self.tree.nodes[ch].outgoing.is_empty() {
                        self.soft_prune_tree(mh);
                    } else {
                        self.tree.moves[mh].pruning = Pruning::None;
                        for &mh in &self.tree.nodes[ch].outgoing {
                            stack.push(mh)
                        }
                    }
                }
            }
        }
    }
    /// Add moves as appropriate.
    fn update_moves(&mut self, nh: NodeHandle) {
        let old_moves = self.tree.nodes[nh]
            .outgoing
            .iter()
            .map(|m| &self.tree.moves[*m].mov)
            .collect::<Vec<&Move<M>>>();
        let current_moves = self.tree.nodes[nh].state.available_moves(&self.mcts);
        let (current_moves, new_moves): (Vec<Move<M>>, Vec<Move<M>>) = current_moves
            .into_iter()
            .partition(|m| old_moves.contains(&&m));
        let (_, failed_moves): (Vec<&Move<M>>, Vec<&Move<M>>) = old_moves
            .into_iter()
            .partition(|m| current_moves.contains(*m));

        let mut pruned = Vec::with_capacity(self.tree.nodes[nh].outgoing.len());
        for &mh in &self.tree.nodes[nh].outgoing {
            if failed_moves.contains(&&self.tree.moves[mh].mov) {
                pruned.push(mh);
            }
        }
        for mh in pruned {
            self.hard_prune_tree(mh);
        }
        for mov in new_moves {
            let handle = self.tree.moves.len();
            let new_move: MoveInfo<M> = MoveInfo {
                handle,
                parent: nh,
                child: None,
                mov,
                pruning: Pruning::None,
            };
            self.tree.moves.push(new_move);
            self.tree.nodes[nh].outgoing.push(handle);
        }
    }
    /// Take a single search step.
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> Result<Vec<NodeHandle>, MCTSError> {
        let (child_state, mh) = self.expand(rng)?;
        let parent_state = &self.tree.nodes[self.tree.moves[mh].parent].state;
        let mov = &self.tree.moves[mh].mov.clone();
        let parents = self.tree.table[parent_state].clone();
        parents
            .iter()
            .filter_map(|ph| {
                self.update_tree(child_state.clone(), *ph, &mov, rng)
                    .ok()
                    .map(|ch| {
                        self.backpropagate_tree(ch);
                        self.soft_prune_tree(mh);
                        Ok(ch)
                    })
            })
            .collect()
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
    fn expand<R: Rng>(&mut self, rng: &mut R) -> Result<(Option<M::State>, MoveHandle), MCTSError> {
        self.can_continue()?;
        let mut nh = self.tree.root;
        for _depth in 0..self.mcts.max_depth() {
            // Choose a move. Avoid moves going to pruned nodes.
            let moves = self.tree.nodes[nh]
                .outgoing
                .iter()
                .copied()
                .filter(|mh| self.tree.moves[*mh].pruning == Pruning::None)
                .map(|mh| &self.tree.moves[mh]);
            // Looks weird, but helps prevent a borrowing issue.
            let mh = self
                .move_eval
                .choose(moves, nh, &self, rng)
                .expect("INVARIANT: active nodes must have moves")
                .handle;
            let mov = &self.tree.moves[mh];
            match mov.child {
                // Descend known moves.
                Some(child_nh) => {
                    nh = child_nh;
                }
                // Take new moves.
                None => {
                    let child_state = self.tree.nodes[nh].state.make_move(
                        nh,
                        &mov.mov,
                        &mut self.mcts,
                        &self.tree,
                    );
                    return Ok((child_state, mh));
                }
            }
        }
        Err(MCTSError::TreeAtMaxDepth)
    }
    fn update_tree<R: Rng>(
        &mut self,
        move_result: Option<M::State>,
        nh: NodeHandle,
        mov: &Move<M>,
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
                self.prune_dag(mh, Pruning::Hard);
                Err(MCTSError::MoveFailed)
            }
            Some(child_state) => {
                let ancestors = self.tree.ancestors_tree(nh);
                let ch = self.make_node(child_state.clone(), rng);
                let entry = self.tree.table.entry(child_state).or_insert_with(|| vec![]);
                // Prevent cycles: don't add nodes whose state is contained in an ancestor.
                if ancestors.iter().any(|a| entry.contains(a)) {
                    self.hard_prune_tree(mh);
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
    fn make_node<R: Rng>(&mut self, s: <M>::State, rng: &mut R) -> NodeHandle {
        let nh = self.tree.nodes.len();
        let mut outgoing = vec![];
        for mov in s.available_moves(&mut self.mcts) {
            let handle = self.tree.moves.len();
            let new_move = MoveInfo {
                handle,
                mov,
                parent: nh,
                child: None,
                pruning: Pruning::None,
            };
            self.tree.moves.push(new_move);
            outgoing.push(handle);
        }
        let evaluation = self.state_eval.evaluate(&s, &mut self.mcts, rng);
        let node = Node {
            state: s,
            incoming: None,
            outgoing,
            evaluation,
            stats: <<M as MCTS>::MoveEval as MoveEvaluator<M>>::NodeStatistics::new(),
        };
        self.tree.nodes.push(node);
        nh
    }
    /// Prunes a `MoveInfo<M>` from the tree.
    fn prune_dag(&mut self, src_mh: MoveHandle, prune: Pruning) {
        let src_mv = &self.tree.moves[src_mh];
        let mov = src_mv.mov.clone();
        let parents = self.tree.table[&self.tree.nodes[src_mv.parent].state].clone();
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
            self.tree.moves[mh].pruning = Pruning::Hard;
            if let Some(nh) = self.tree.moves[mh].child {
                self.tree.delist(nh);
                for &omh in &self.tree.nodes[nh].outgoing {
                    if self.tree.moves[omh].pruning != Pruning::Hard {
                        stack.push(omh);
                    }
                }
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

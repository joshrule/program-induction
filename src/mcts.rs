//! Representations capable of [Monte Carlo Tree Search].
//!
//! [Monte Carlo Tree Search]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

// TODO:
// - Find States via Hash rather than Eq.
// - ask for moves & states to be Debug for easier debugging.
// - Create a recycling pool for moves & nodes.
use rand::Rng;
use serde_json::Value;

type Adjust<M> = <<M as MCTS>::State as State<M>>::AbstractDepthAdjustment;
type Move<M> = <<M as MCTS>::State as State<M>>::Move;
type StateEvaluation<M> = <<M as MCTS>::StateEval as StateEvaluator<M>>::StateEvaluation;
pub type NodeHandle = usize;
pub type MoveHandle = usize;

pub trait State<M: MCTS<State = Self>>: std::hash::Hash + Eq + Sized + Sync {
    type Move: std::fmt::Display + Clone + Sync + Send;
    type MoveList: IntoIterator<Item = Self::Move>;
    type AbstractDepthAdjustment;
    fn available_moves(&self, mcts: &mut M) -> Self::MoveList;
    fn make_move<R: Rng>(
        &self,
        mov: &Self::Move,
        mcts: &mut M,
        rng: &mut R,
    ) -> Option<(Self, Option<Self::AbstractDepthAdjustment>)>;
    fn add_moves_for_new_data(&self, moves: &[Self::Move], mcts: &mut M) -> Self::MoveList;
    fn adjust_abstract_depth(&self, adjustment: &Self::AbstractDepthAdjustment, mcts: &mut M);
    fn describe_self(&self, mcts: &M) -> Value;
    fn describe_move(&self, mv: &Self::Move, mcts: &M) -> Value;
}

pub trait MoveEvaluator<M: MCTS<MoveEval = Self>>: Sized + Sync {
    type MoveEvaluation: Clone + Sync + Send;
    fn choose<'a, MoveIter>(&self, MoveIter, NodeHandle, &SearchTree<M>) -> Option<&'a MoveInfo<M>>
    where
        MoveIter: Iterator<Item = &'a MoveInfo<M>>;
}

pub trait StateEvaluator<M: MCTS<StateEval = Self>>: Sized + Sync {
    type StateEvaluation: Clone + Sync + Send + Into<f64>;
    fn evaluate<R: Rng>(&self, &M::State, &mut M, &mut R) -> Self::StateEvaluation;
    fn reread(&self, &M::State, &mut M) -> Self::StateEvaluation;
    fn zero(&self) -> f64 {
        0.0
    }
}

pub trait MCTS: Sized + Sync {
    type StateEval: StateEvaluator<Self>;
    type MoveEval: MoveEvaluator<Self>;
    type State: State<Self>;
    fn max_depth(&self) -> usize {
        std::usize::MAX
    }
    fn max_states(&self) -> usize {
        std::usize::MAX
    }
    fn combine_qs(&self, q1: f64, q2: f64) -> f64 {
        q1 + q2
    }
}

pub struct MCTSManager<M: MCTS> {
    tree: SearchTree<M>,
}

pub struct SearchTree<M: MCTS> {
    mcts: M,
    root: NodeHandle,
    nodes: Vec<Node<M>>,
    moves: Vec<MoveInfo<M>>,
    state_eval: M::StateEval,
    move_eval: M::MoveEval,
}

pub struct Node<M: MCTS> {
    state: M::State,
    incoming: Vec<MoveHandle>,
    outgoing: Vec<MoveHandle>,
    evaluation: StateEvaluation<M>,
    maximum_valid_depth: usize,
    soft_pruned: bool,
    pub q: f64,
    pub n: f64,
}

pub struct MoveInfo<M: MCTS> {
    handle: MoveHandle,
    parent: NodeHandle,
    pub child: Option<NodeHandle>,
    pub mov: Move<M>,
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

impl<M: MCTS> Node<M> {
    fn show(&self) {
        println!("  incoming: {:?}", self.incoming);
        println!("  outgoing: {:?}", self.outgoing);
        println!("  q/n: {:.4}/{:.4}", self.q, self.n);
        println!("  maximum_valid_depth: {}", self.maximum_valid_depth);
        println!("  soft_pruned: {}", self.soft_pruned);
    }
    pub fn state(&self) -> &M::State {
        &self.state
    }
}

impl<M: MCTS> MoveInfo<M> {
    fn show(&self) {
        println!("  handle: {}", self.handle);
        println!("  parent: {}", self.parent);
        println!("  child: {:?}", self.child);
        println!("  move: {}", self.mov);
    }
}

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
        let mut steps = 0;
        while !predicate(&self.tree.mcts) {
            match self.tree.step(rng) {
                Ok(nh) => {
                    println!("# step reached {}", nh);
                    steps += 1;
                }
                Err(e) => {
                    println!("# {}", e);
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
        println!("# ended after steps: {}", steps);
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
            })
            .collect();
        let root_node = Node {
            state: root,
            incoming: vec![],
            outgoing: (0..moves.len()).collect(),
            q: evaluation.clone().into(),
            n: 1.0,
            evaluation,
            maximum_valid_depth: mcts.max_depth() - (moves.is_empty() as usize),
            soft_pruned: moves.is_empty(),
        };
        SearchTree {
            mcts,
            state_eval,
            move_eval,
            moves,
            root: 0,
            nodes: vec![root_node],
        }
    }
    pub fn summary(&self) -> (usize, usize) {
        (self.nodes.len(), self.moves.len())
    }
    pub fn show(&self) {
        println!("SearchTree");
        println!("root: {}", self.root);
        println!();
        println!("nodes: {}", self.nodes.len());
        for (i, node) in self.nodes.iter().enumerate() {
            println!("{}", i);
            node.show()
        }
        println!();
        println!("moves: {}", self.moves.len());
        for (i, mov) in self.moves.iter().enumerate() {
            println!("{}", i);
            mov.show()
        }
    }
    pub fn to_file(&self, data_file: &str) -> std::io::Result<()> {
        let moves = self
            .moves
            .iter()
            .enumerate()
            .map(|(i, mv)| {
                json!({
                    "handle": i,
                    "parent": mv.parent,
                    "child": mv.child,
                    "move": self.nodes[mv.parent].state.describe_move(&mv.mov, &self.mcts),
                })
            })
            .collect::<Vec<_>>();
        let nodes = self
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
                    "q": n.q,
                    "n": n.n,
                    "mvd": n.maximum_valid_depth,
                    "pruned": n.soft_pruned,
                })
            })
            .collect::<Vec<_>>();
        let tree = json!({
            "root": self.root,
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
    pub fn node(&self, node: NodeHandle) -> &Node<M> {
        &self.nodes[node]
    }
    pub fn mov(&self, mov: MoveHandle) -> &MoveInfo<M> {
        &self.moves[mov]
    }
    pub fn reevaluate_states(&mut self) {
        println!("#     reevaluating states");
        // iterate over the nodes and copy the evaluation information from the objects.
        println!("#       reevaluating nodes");
        for node in self.nodes.iter_mut() {
            node.evaluation = self.state_eval.reread(&node.state, &mut self.mcts);
        }
        println!("#       updating qs");
        self.recompute_qs();
    }
    pub fn recompute_qs(&mut self) {
        let mut stack = vec![self.root];
        let mut finished = vec![];
        while let Some(&node) = stack.last() {
            let ready = self.nodes[node]
                .outgoing
                .iter()
                .all(|&mh| match self.moves[mh].child {
                    None => true,
                    Some(nh) => finished.contains(&nh),
                });
            if ready {
                let mut new_q = self.nodes[node].evaluation.clone().into();
                for &mh in &self.nodes[node].outgoing {
                    if let Some(child) = self.moves[mh].child {
                        new_q = self.mcts.combine_qs(new_q, self.nodes[child].q);
                    }
                }
                self.nodes[node].q = new_q;
                finished.push(stack.pop().expect("INVARIANT: stack has last element"));
            } else {
                for &mh in &self.nodes[node].outgoing {
                    if let Some(nh) = self.moves[mh].child {
                        stack.push(nh);
                    }
                }
            }
        }
    }
    pub fn update_moves(&mut self) {
        let mut unpruned = Vec::with_capacity(self.nodes.len());
        // Add moves as appropriate.
        for nh in 0..self.nodes.len() {
            let old_moves = self.nodes[nh]
                .outgoing
                .iter()
                .map(|&m| &self.moves[m].mov)
                .cloned()
                .collect::<Vec<_>>();
            for mov in self.nodes[nh]
                .state
                .add_moves_for_new_data(&old_moves, &mut self.mcts)
            {
                let handle = self.moves.len();
                let new_move = MoveInfo {
                    handle,
                    mov,
                    parent: nh,
                    child: None,
                };
                self.moves.push(new_move);
                self.nodes[nh].outgoing.push(handle);
                unpruned.push(nh);
            }
        }
        // Backpropagate any changes in soft pruning.
        while let Some(nh) = unpruned.pop() {
            if self.nodes[nh].soft_pruned {
                println!("#       unpruning {}", nh);
            }
            self.nodes[nh].soft_pruned = false;
            for &mh in &self.nodes[nh].incoming {
                let parent = self.moves[mh].parent;
                if !unpruned.contains(&parent) {
                    unpruned.push(parent);
                }
            }
        }
    }
    /// Take a single search step.
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> Result<NodeHandle, MCTSError> {
        self.expand(rng).map(|nh| {
            self.backpropagate(nh);
            self.soft_prune(nh);
            self.update_maximum_valid_depths(nh);
            nh
        })
    }
    // Expand the search tree by taking a single move.
    fn expand<R: Rng>(&mut self, rng: &mut R) -> Result<NodeHandle, MCTSError> {
        if self.nodes.len() >= self.mcts.max_states() {
            return Err(MCTSError::TreeAtMaxStates);
        } else if self.nodes[self.root].soft_pruned {
            return Err(MCTSError::TreeExhausted);
        } else if self.mcts.max_depth() == 0
            || self.nodes[self.root]
                .outgoing
                .iter()
                .all(|mh| match self.moves[*mh].child {
                    Some(nh) => self.nodes[nh].maximum_valid_depth == 0,
                    None => false,
                })
        {
            return Err(MCTSError::TreeAtMaxDepth);
        }
        let mut nh = self.root;
        for depth in 0..self.mcts.max_depth() {
            // Choose a move.
            let moves = self.nodes[nh]
                .outgoing
                .iter()
                // Avoid moves going too deep or to pruned nodes.
                .filter(|mh| match self.moves[**mh].child {
                    None => true,
                    Some(child_nh) => {
                        !self.nodes[child_nh].soft_pruned
                            && self.nodes[child_nh].maximum_valid_depth > depth
                    }
                })
                .map(|mh| &self.moves[*mh]);
            println!("#   got some moves");
            // Looks weird, but helps prevent a borrowing issue.
            let mh = self
                .move_eval
                .choose(moves, nh, &self)
                .expect("INVARIANT: active nodes must have moves")
                .handle;
            let mov = &self.moves[mh];
            match mov.child {
                // Descend known moves.
                Some(child_nh) => {
                    nh = child_nh;
                    println!("#   Child is {} at depth {}.", child_nh, depth + 1);
                }
                // Take new moves.
                None => {
                    println!("#   No child. Taking the move at depth {}.", depth);
                    let child_state = self.nodes[nh]
                        .state
                        .make_move(&mov.mov, &mut self.mcts, rng);
                    println!("#   Updating tree.");
                    return self.update_tree(child_state, mh, rng);
                }
            }
        }
        Err(MCTSError::TreeAtMaxDepth)
    }
    fn update_tree<R: Rng>(
        &mut self,
        move_result: Option<(<M>::State, Option<Adjust<M>>)>,
        mov: MoveHandle,
        rng: &mut R,
    ) -> Result<NodeHandle, MCTSError> {
        match move_result {
            None => {
                self.hard_prune(mov);
                Err(MCTSError::MoveFailed)
            }
            Some((child_state, adjustment)) => {
                // Identify the parent handle and child handle.
                let ph = self.moves[mov].parent;
                let ch = self.find_or_make_node(child_state, rng);
                // Prevent cycles: don't add moves that make a child its own ancestor.
                if self.ancestors(ph).contains(&ch) || ph == ch {
                    println!(
                        "#     Cycle detected! Ignoring the child ({} {}).",
                        self.ancestors(ph).contains(&ch),
                        ph == ch
                    );
                    self.hard_prune(mov);
                    Err(MCTSError::MoveCreatedCycle)
                } else {
                    // Give the move a child.
                    self.moves[mov].child = Some(ch);
                    // Add the move as a parent of the child, if new.
                    if !self.nodes[ch].incoming.contains(&mov) {
                        self.nodes[ch].incoming.push(mov);
                    }
                    self.adjust_abstract_depth(ch, &adjustment);
                    Ok(ch)
                }
            }
        }
    }
    fn hard_prune(&mut self, mh: MoveHandle) {
        println!("#       hard pruning {}", mh);
        // Remove the move from the parent's outgoing list.
        let parent = self.moves[mh].parent;
        self.nodes[parent].outgoing = self.nodes[parent]
            .outgoing
            .iter()
            .filter(|omh| **omh != mh)
            .copied()
            .collect();
        // Keep the tree consistent.
        self.soft_prune(parent);
        self.update_maximum_valid_depths(parent);
    }
    fn soft_prune(&mut self, start: NodeHandle) {
        let mut stack = vec![start];
        while let Some(nh) = stack.pop() {
            // Nodes are soft-pruned if they have no unexplored descendants.
            let prunable = self.nodes[nh]
                .outgoing
                .iter()
                .all(|mh| match self.moves[*mh].child {
                    None => false,
                    Some(child_nh) => self.nodes[child_nh].soft_pruned,
                });
            if prunable {
                println!("#       soft pruning {}", nh);
                self.nodes[nh].soft_pruned = true;
            }
            // Add nodes to stack as appropriate.
            for &mh in &self.nodes[nh].incoming {
                stack.push(self.moves[mh].parent)
            }
        }
    }
    fn find_or_make_node<R: Rng>(&mut self, s: <M>::State, rng: &mut R) -> NodeHandle {
        match self.nodes.iter().position(|n| n.state == s) {
            Some(h) => {
                println!("          found node {}", h);
                h
            }
            None => {
                let node_handle = self.nodes.len();
                let evaluation = self.state_eval.evaluate(&s, &mut self.mcts, rng);
                let incoming = vec![];
                let mut outgoing = vec![];
                for mov in s.available_moves(&mut self.mcts) {
                    println!("#     pushing move: {}", mov);
                    let handle = self.moves.len();
                    let new_move = MoveInfo {
                        handle,
                        mov,
                        parent: node_handle,
                        child: None,
                    };
                    self.moves.push(new_move);
                    outgoing.push(handle);
                }
                let node = Node {
                    state: s,
                    n: 0.0, // fixed during backprop
                    maximum_valid_depth: self.mcts.max_depth() - (outgoing.is_empty() as usize),
                    soft_pruned: outgoing.is_empty(),
                    q: self.state_eval.zero(),
                    evaluation,
                    incoming,
                    outgoing,
                };
                println!(
                    "#     made node {} with {} outgoing moves",
                    node_handle,
                    node.outgoing.len()
                );
                self.nodes.push(node);
                node_handle
            }
        }
    }
    // Bubble search statistics and pruning information back to the root.
    fn backpropagate(&mut self, new_node: NodeHandle) {
        println!("#   backpropagating from {}.", new_node);
        let evaluation = self.nodes[new_node].evaluation.clone().into();
        let mut stack = vec![new_node];
        while let Some(nh) = stack.pop() {
            println!("#     dealing with node {}.", nh);
            // Update q.
            self.nodes[nh].q = self.mcts.combine_qs(self.nodes[nh].q, evaluation);
            // Update n.
            self.nodes[nh].n += 1.0;
            // Add nodes to stack as appropriate.
            for &mh in &self.nodes[nh].incoming {
                stack.push(self.moves[mh].parent)
            }
        }
    }
    fn update_maximum_valid_depths(&mut self, new_node: NodeHandle) {
        let mut stack = vec![new_node];
        while let Some(nh) = stack.pop() {
            let current_depth = self.nodes[nh].maximum_valid_depth;
            // If nh has children, max_valid_depth <-
            //    max(max_valid_depth(child) for child in children if !child.soft_pruned) - 1.
            // else max_valid_depth <- depth_limit
            let new_depth = self.nodes[nh]
                .outgoing
                .iter()
                .map(|m| {
                    self.moves[*m]
                        .child
                        .filter(|ch| !self.nodes[*ch].soft_pruned)
                        .map(|ch| self.nodes[ch].maximum_valid_depth)
                        .unwrap_or_else(|| self.mcts.max_depth())
                })
                .max()
                .map(|max_max| max_max.saturating_sub(1))
                .unwrap_or_else(|| self.mcts.max_depth());
            if new_depth != current_depth {
                // Update the maximum valid depth.
                self.nodes[nh].maximum_valid_depth = new_depth;
                // Add moves leading to the parent to the stack.
                for &mh in &self.nodes[nh].incoming {
                    stack.push(self.moves[mh].parent);
                }
            }
        }
    }
    fn adjust_abstract_depth(&mut self, nh: NodeHandle, adjustment: &Option<Adjust<M>>) {
        match *adjustment {
            None => (),
            Some(ref adjustment) => {
                self.nodes[nh]
                    .state
                    .adjust_abstract_depth(adjustment, &mut self.mcts);
                for dh in self.descendants(nh) {
                    self.nodes[dh]
                        .state
                        .adjust_abstract_depth(adjustment, &mut self.mcts);
                }
            }
        }
    }
    fn ancestors(&self, node: NodeHandle) -> Vec<NodeHandle> {
        let mut stack = vec![node];
        let mut ancestors = vec![];
        while let Some(node) = stack.pop() {
            for &mov in &self.nodes[node].incoming {
                let parent = self.moves[mov].parent;
                if !ancestors.contains(&parent) {
                    ancestors.push(parent);
                    stack.push(parent);
                }
            }
        }
        ancestors
    }
    fn descendants(&self, node: NodeHandle) -> Vec<NodeHandle> {
        let mut stack = vec![node];
        let mut descendants = vec![];
        while let Some(node) = stack.pop() {
            for &mh in &self.nodes[node].outgoing {
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
}

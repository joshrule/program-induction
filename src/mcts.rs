//! Representations capable of [Monte Carlo Tree Search].
//!
//! [Monte Carlo Tree Search]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

// TODO:
// - Replace Eq constraint on State with some sort of hash for speed.
// - add transposition tables as a configurable feature
//   - Define the trait.
//     pub trait TranspositionTable<M: MCTS> {
//         fn insert(&self, key: &M::State, value: Node<M>);
//         fn get(&self, key: &M::State) -> Option<&Node<M>>;
//     }
//   - Add to MCTS as an associated type.
//   - Replace SearchTree nodes with a table.

type Move<M> = <<M as MCTS>::State as State>::Move;
type StateEvaluation<M> = <<M as MCTS>::StateEval as StateEvaluator<M>>::StateEvaluation;
type NodeHandle = usize;
type MoveHandle = usize;

pub trait State: Eq + Sized + Sync {
    type Move: Clone + Sync + Send;
    type MoveList: IntoIterator<Item = Self::Move>;
    fn available_moves(&self) -> Self::MoveList;
    fn make_move(&self, mov: &Self::Move) -> Vec<Self>;
    fn uniquify(taken: &Self::Move, generated: usize) -> (Vec<Self::Move>, Vec<Self::Move>);
}

pub trait MoveEvaluator<M: MCTS<MoveEval = Self>>: Sized + Sync {
    type MoveEvaluation: Clone + Sync + Send;
    fn choose<'a, MoveIter>(&self, MoveIter) -> Option<&'a MoveInfo<M>>
    where
        MoveIter: Iterator<Item = &'a MoveInfo<M>>;
    // Update the available moves given that we took this one.
    // - add moves as necessary (we never remove them)
    // - re-evaluate moves as necessary
    fn update_moves(&self, &mut Vec<MoveInfo<M>>, &MoveInfo<M>);
}

pub trait StateEvaluator<M: MCTS<StateEval = Self>>: Sized + Sync {
    type StateEvaluation: Clone + Sync + Send + Into<f64>;
    fn evaluate(&self, &M::State) -> Self::StateEvaluation;
}

pub trait MCTS: Sized + Sync {
    type StateEval: StateEvaluator<Self>;
    type MoveEval: MoveEvaluator<Self>;
    type State: State;
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
    root: usize,
    nodes: Vec<Node<M>>,
    moves: Vec<MoveInfo<M>>,
    size: usize,
    state_eval: M::StateEval,
    move_eval: M::MoveEval,
}

pub struct Node<M: MCTS> {
    state: M::State,
    incoming: Vec<MoveHandle>,
    outgoing: Vec<MoveHandle>,
    evaluation: StateEvaluation<M>,
    q: f64,
    n: f64,
}

pub struct MoveInfo<M: MCTS> {
    handle: MoveHandle,
    parent: NodeHandle,
    child: Option<NodeHandle>,
    mov: Move<M>,
    maximum_valid_depth: usize,
}

impl<M: MCTS> MCTSManager<M> {
    // Search until the predicate evaluates to `true`.
    pub fn step_until<P: Fn(&M) -> bool>(&mut self, predicate: P) {
        while !predicate(&self.tree.mcts) {
            self.tree.step();
        }
    }
    // Take a single search step.
    pub fn step(&mut self) {
        self.tree.step();
    }
}

impl<M: MCTS> SearchTree<M> {
    /// Take a single search step.
    pub fn step(&mut self) -> usize {
        match self.expand() {
            Some((path, moves)) => {
                self.size += moves.len();
                for &m in &moves {
                    // INVARIANT: Moves returned by expand always have children.
                    self.backpropagate(self.moves[m].child.unwrap(), &path);
                    self.update_maximum_valid_depths(m);
                }
                moves.len()
            }
            _ => 0,
        }
    }
    // Expand the search tree by taking a single move.
    fn expand(&mut self) -> Option<(Vec<NodeHandle>, Vec<MoveHandle>)> {
        if self.nodes.len() >= self.mcts.max_states() {
            return None;
        }
        let mut path = vec![self.root];
        loop {
            // Choose a move. INVARIANT: path always has at least 1 element.
            let handle = *path.last().unwrap();
            let moves = self.nodes[handle].outgoing.iter().map(|n| &self.moves[*n]);
            let mov: MoveHandle = self.move_eval.choose(moves)?.handle;
            let child = self.moves[mov].child;
            match child {
                // Descend known moves.
                Some(handle) => path.push(handle),
                // Take new moves.
                None => {
                    let child_states = self.nodes[handle].state.make_move(&self.moves[mov].mov);
                    let (child_moves, new_moves) =
                        <M>::State::uniquify(&self.moves[mov].mov, child_states.len());
                    let moves_to_backprop =
                        self.update_tree(child_states, child_moves, new_moves, mov);
                    return Some((path, moves_to_backprop));
                }
            }
        }
    }
    // Add `MoveInfo`s representing each of the newly created `child_states`.
    fn update_tree(
        &mut self,
        mut child_states: Vec<<M>::State>,
        mut child_moves: Vec<Move<M>>,
        new_moves: Vec<Move<M>>,
        mov: MoveHandle,
    ) -> Vec<MoveHandle> {
        // If the move failed, don't change anything.
        if child_states.is_empty() {
            vec![]
        } else {
            // Splice in newly available moves (new_moves).
            let parent = self.moves[mov].parent;
            let mut new_handles = vec![];
            for mov in new_moves {
                let handle = self.moves.len();
                let new_move = MoveInfo {
                    handle,
                    parent,
                    mov,
                    child: None,
                    maximum_valid_depth: self.mcts.max_depth(),
                };
                self.moves.push(new_move);
                new_handles.push(handle);
            }
            self.nodes[parent].outgoing.append(&mut new_handles);
            // Reuse the initial move by giving it the first child.
            let mut handles = vec![];
            while !child_states.is_empty() {
                let first_state = child_states.swap_remove(0);
                let first_move = child_moves.swap_remove(0);
                if let Some(handle) = self.make_first_child(first_state, first_move, mov) {
                    handles.push(handle);
                    break;
                }
            }
            if handles.is_empty() {
                vec![]
            } else {
                // Process the remaining moves.
                let mut rest = child_states
                    .into_iter()
                    .zip(child_moves)
                    .filter_map(|(s, m)| self.make_child(s, m, self.moves[mov].parent))
                    .collect();
                handles.append(&mut rest);
                handles
            }
        }
    }
    fn make_first_child(&mut self, s: <M>::State, m: Move<M>, h: MoveHandle) -> Option<MoveHandle> {
        // Update the move.
        let node = self.find_or_make_node(s, h);
        // Eliminate cycles. Don't add a move if the child is the ancestor of the parent.
        if self.ancestors(node).contains(&self.moves[h].parent) {
            None
        } else {
            let mov = &mut self.moves[h];
            mov.child = Some(node);
            mov.mov = m;
            // The maximum_valid_depth is examined during the update.
            // Return the handle.
            Some(h)
        }
    }
    // For each state `s`, add the appropriate moves to the graph.
    fn make_child(&mut self, s: <M>::State, m: Move<M>, parent: NodeHandle) -> Option<MoveHandle> {
        // Create the move.
        let handle = self.moves.len();
        let node = self.find_or_make_node(s, handle);
        // Eliminate cycles. Don't add a move if the child is the ancestor of the parent.
        if self.ancestors(node).contains(&parent) {
            None
        } else {
            let new_move = MoveInfo {
                handle,
                parent,
                child: Some(node),
                mov: m,
                // The maximum_valid_depth is incorrect but gets fixed during the update.
                maximum_valid_depth: self.mcts.max_depth(),
            };
            // Add it to the tree's list of moves.
            self.moves.push(new_move);
            // Return the handle.
            Some(handle)
        }
    }
    fn find_or_make_node(&mut self, s: <M>::State, incoming: MoveHandle) -> NodeHandle {
        match self.nodes.iter().position(|n| n.state == s) {
            Some(h) => h,
            None => {
                let node_handle = self.nodes.len();
                let evaluation = self.state_eval.evaluate(&s);
                let q: f64 = evaluation.clone().into();
                let mut outgoing = vec![];
                for m in s.available_moves() {
                    let handle = self.moves.len();
                    let new_move = MoveInfo {
                        handle,
                        parent: node_handle,
                        child: None,
                        mov: m.clone(),
                        maximum_valid_depth: self.mcts.max_depth(),
                    };
                    self.moves.push(new_move);
                    outgoing.push(handle);
                }
                let node = Node {
                    incoming: vec![incoming],
                    state: s,
                    n: 1.0,
                    outgoing,
                    evaluation,
                    q,
                };
                self.nodes.push(node);
                node_handle
            }
        }
    }
    // Bubble search statistics back to the root.
    fn backpropagate(&mut self, new_node: NodeHandle, path: &[NodeHandle]) {
        let evaluation = self.nodes[new_node].evaluation.clone().into();
        for &handle in path {
            self.nodes[handle].n += 1.0;
            self.nodes[handle].q += evaluation;
        }
    }
    // Invalidate fully explored nodes, i.e. moves whose child has no valid moves.
    fn update_maximum_valid_depths(&mut self, final_move: MoveHandle) {
        let mut stack = vec![final_move];
        while let Some(mov) = stack.pop() {
            let current_depth = self.moves[mov].maximum_valid_depth;
            let new_depth = match self.moves[mov].child {
                Some(node) => {
                    // If m has children, it's maximum valid depth is max(max_valid_depth(child) for child in children) - 1.
                    let max_max = self.nodes[node]
                        .outgoing
                        .iter()
                        .map(|m| self.moves[*m].maximum_valid_depth)
                        .max();
                    match max_max {
                        Some(max) => max.saturating_sub(1),
                        None => 0,
                    }
                }
                None => self.mcts.max_depth(),
            };
            if new_depth != current_depth {
                // Update the maximum valid depth.
                self.moves[mov].maximum_valid_depth = new_depth;
                // Add moves leading to the parent to the stack.
                for affected_move in &self.nodes[self.moves[mov].parent].incoming {
                    stack.push(*affected_move);
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
}

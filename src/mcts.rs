//! Representations capable of [Monte Carlo Tree Search].
//!
//! [Monte Carlo Tree Search]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

// TODO:
// - Replace Eq constraint on State with some sort of hash for speed.
// - ask for moves & states to be Debug for better debugging.
// - add transposition tables as a configurable feature
//   - Define the trait.
//     pub trait TranspositionTable<M: MCTS> {
//         fn insert(&self, key: &M::State, value: Node<M>);
//         fn get(&self, key: &M::State) -> Option<&Node<M>>;
//     }
//   - Add to MCTS as an associated type.
//   - Replace SearchTree nodes with a table.
use rand::Rng;

type Move<M> = <<M as MCTS>::State as State<M>>::Move;
type StateEvaluation<M> = <<M as MCTS>::StateEval as StateEvaluator<M>>::StateEvaluation;
pub type NodeHandle = usize;
pub type MoveHandle = usize;

pub trait State<M: MCTS<State = Self>>: PartialEq + Sized + Sync {
    type Move: Clone + Sync + Send;
    type MoveList: IntoIterator<Item = Self::Move>;
    fn available_moves(&self, mcts: &M) -> Self::MoveList;
    fn make_move<R: Rng>(&self, mov: &Self::Move, mcts: &mut M, rng: &mut R) -> Vec<Self>;
    fn uniquify(taken: &Self::Move, generated: usize) -> (Vec<Self::Move>, Vec<Self::Move>);
    fn add_moves_for_new_data(&self, moves: &[(&Self::Move, bool)], mcts: &M) -> Self::MoveList;
}

pub trait MoveEvaluator<M: MCTS<MoveEval = Self>>: Sized + Sync {
    type MoveEvaluation: Clone + Sync + Send;
    fn choose<'a, MoveIter>(&self, MoveIter, NodeHandle, &SearchTree<M>) -> Option<&'a MoveInfo<M>>
    where
        MoveIter: Iterator<Item = &'a MoveInfo<M>>;
}

pub trait StateEvaluator<M: MCTS<StateEval = Self>>: Sized + Sync {
    type StateEvaluation: Clone + Sync + Send + Into<f64>;
    fn evaluate(&self, &M::State, &M) -> Self::StateEvaluation;
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
    size: usize,
    state_eval: M::StateEval,
    move_eval: M::MoveEval,
}

pub struct Node<M: MCTS> {
    state: M::State,
    incoming: Vec<MoveHandle>,
    outgoing: Vec<MoveHandle>,
    evaluation: StateEvaluation<M>,
    pub q: f64,
    pub n: f64,
}

pub struct MoveInfo<M: MCTS> {
    handle: MoveHandle,
    parent: NodeHandle,
    pub child: Option<NodeHandle>,
    pub mov: Move<M>,
    maximum_valid_depth: usize,
}

impl<M: MCTS> Node<M> {
    fn show(&self) {
        println!("incoming: {:?}", self.incoming);
        println!("outgoing: {:?}", self.outgoing);
        println!("q/n: {:.4}/{:.4}", self.q, self.n);
    }
}

impl<M: MCTS> MoveInfo<M> {
    fn show(&self) {
        println!("handle: {}", self.handle);
        println!("parent: {}", self.parent);
        println!("child: {:?}", self.child);
        println!("maximum_valid_depth: {}", self.maximum_valid_depth);
    }
}

impl<M: MCTS> MCTSManager<M> {
    pub fn new(mcts: M, root: M::State, state_eval: M::StateEval, move_eval: M::MoveEval) -> Self {
        let tree = SearchTree::new(mcts, root, state_eval, move_eval);
        MCTSManager { tree }
    }
    // Search until the predicate evaluates to `true`.
    pub fn step_until<R: Rng, P: Fn(&M) -> bool>(&mut self, rng: &mut R, predicate: P) {
        let mut steps = 0;
        let mut result = Some(0);
        while result.is_some() && !predicate(&self.tree.mcts) {
            result = self.tree.step(rng);
            steps += 1;
        }
        println!("steps: {}", steps);
    }
    // Take a single search step.
    pub fn step<R: Rng>(&mut self, rng: &mut R) {
        self.tree.step(rng);
    }
    pub fn tree(&self) -> &SearchTree<M> {
        &self.tree
    }
    pub fn tree_mut(&mut self) -> &mut SearchTree<M> {
        &mut self.tree
    }
}

impl<M: MCTS> SearchTree<M> {
    pub fn new(mcts: M, root: M::State, state_eval: M::StateEval, move_eval: M::MoveEval) -> Self {
        let evaluation = state_eval.evaluate(&root, &mcts);
        let moves: Vec<_> = root
            .available_moves(&mcts)
            .into_iter()
            .enumerate()
            .map(|(i, mov)| MoveInfo {
                handle: i,
                parent: 0,
                mov,
                child: None,
                maximum_valid_depth: mcts.max_depth(),
            })
            .collect();
        let root_node = Node {
            state: root,
            incoming: vec![],
            outgoing: (0..moves.len()).collect(),
            q: evaluation.clone().into(),
            n: 1.0,
            evaluation,
        };
        SearchTree {
            mcts,
            state_eval,
            move_eval,
            moves,
            root: 0,
            size: 1,
            nodes: vec![root_node],
        }
    }
    pub fn show(&self) {
        println!("SearchTree");
        println!("size: {}", self.size);
        println!("root: {}", self.root);
        println!("nodes: {}", self.nodes.len());
        for (i, node) in self.nodes.iter().enumerate() {
            println!("{}", i);
            node.show()
        }
        println!("moves: {}", self.moves.len());
        for (i, mov) in self.moves.iter().enumerate() {
            println!("{}", i);
            mov.show()
        }
    }
    pub fn mcts(&self) -> &M {
        &self.mcts
    }
    pub fn node(&self, node: NodeHandle) -> &Node<M> {
        &self.nodes[node]
    }
    pub fn mov(&self, mov: MoveHandle) -> &MoveInfo<M> {
        &self.moves[mov]
    }
    pub fn mcts_mut(&mut self) -> &mut M {
        &mut self.mcts
    }
    pub fn reevaluate_states(&mut self) {
        // iterate over the nodes and copy the evaluation information from the objects.
        for node in self.nodes.iter_mut() {
            node.evaluation = self.state_eval.evaluate(&node.state, &self.mcts);
            println!(
                "#     new evaluation: {:.4}",
                node.evaluation.clone().into()
            );
        }
        println!("#   reevaluated nodes");
        self.recompute_qs();
        println!("#   updated qs");
    }
    pub fn recompute_qs(&mut self) {
        let mut stack = vec![self.root];
        let mut finished = vec![];
        while let Some(&node) = stack.last() {
            println!("#      stack: {:?}", stack);
            println!("#      finished: {:?}", finished);
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
        for nh in 0..self.nodes.len() {
            let old_moves = self.nodes[nh]
                .outgoing
                .iter()
                .map(|&m| (&self.moves[m].mov, self.moves[m].child.is_some()))
                .collect::<Vec<_>>();
            for mov in self.nodes[nh]
                .state
                .add_moves_for_new_data(&old_moves, &self.mcts)
            {
                let handle = self.moves.len();
                let new_move = MoveInfo {
                    handle,
                    mov,
                    parent: nh,
                    child: None,
                    maximum_valid_depth: self.mcts.max_depth(),
                };
                self.moves.push(new_move);
                self.nodes[nh].outgoing.push(handle);
            }
        }
    }
    /// Take a single search step.
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> Option<usize> {
        println!(
            "  Searching with {} nodes and {} moves",
            self.nodes.len(),
            self.moves.len()
        );
        match self.expand(rng) {
            Some((path, moves)) => {
                println!(
                    "  Search succeeded: {} new moves with a path length of {}.",
                    moves.len(),
                    path.len(),
                );
                self.size += moves.len();
                for &m in &moves {
                    // INVARIANT: Moves returned by expand always have children.
                    self.backpropagate(self.moves[m].child.unwrap(), &path);
                    self.update_maximum_valid_depths(m);
                }
                Some(moves.len())
            }
            _ => {
                println!("  Search failed");
                None
            }
        }
    }
    // Expand the search tree by taking a single move.
    fn expand<R: Rng>(&mut self, rng: &mut R) -> Option<(Vec<NodeHandle>, Vec<MoveHandle>)> {
        if self.nodes.len() >= self.mcts.max_states() {
            println!(
                "too many nodes: {} >= {}",
                self.nodes.len(),
                self.mcts.max_states()
            );
            return None;
        }
        let mut path = vec![self.root];
        loop {
            // Choose a move. INVARIANT: path always has at least 1 element.
            let handle = *path.last().unwrap();
            let moves = self.nodes[handle].outgoing.iter().map(|n| &self.moves[*n]);
            println!("    got some moves");
            let mov: MoveHandle = self.move_eval.choose(moves, handle, &self)?.handle;
            println!("    chose a move");
            let child = self.moves[mov].child;
            println!("    found the child: {:?}", child);
            match child {
                // Descend known moves.
                Some(handle) => {
                    println!("    descending");
                    path.push(handle);
                }
                // Take new moves.
                None => {
                    println!("    taking the move");
                    let child_states = self.nodes[handle].state.make_move(
                        &self.moves[mov].mov,
                        &mut self.mcts,
                        rng,
                    );
                    println!("    created {} child states", child_states.len());
                    let (new_moves, child_moves) =
                        <M>::State::uniquify(&self.moves[mov].mov, child_states.len());
                    println!("    uniquified: {} {}", child_moves.len(), new_moves.len());
                    let moves_to_backprop =
                        self.update_tree(child_states, child_moves, new_moves, mov);
                    println!("    updated tree and returning");
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
        mut new_moves: Vec<Move<M>>,
        mov: MoveHandle,
    ) -> Vec<MoveHandle> {
        let ph = self.moves[mov].parent;
        match (!child_moves.is_empty(), !new_moves.is_empty()) {
            // We have no new states.
            (false, false) => {
                println!("      no new states to handle");
                // Unlink the move. TODO: put the move in a recycling pool.
                let parent = &mut self.nodes[ph];
                let idx = parent.outgoing.iter().position(|&mh| mh == mov);
                if let Some(idx) = idx {
                    parent.outgoing.swap_remove(idx);
                }
                vec![]
            }
            // We only have new moves.
            (false, true) => {
                // Recycle the old move with a new move.
                println!(
                    "      parent has {} outgoing moves",
                    self.nodes[ph].outgoing.len()
                );
                println!("        recycled a move");
                let first_move = new_moves.swap_remove(0);
                self.moves[mov].mov = first_move;
                for mov in new_moves {
                    println!("        added a new move");
                    let handle = self.moves.len();
                    let new_move = MoveInfo {
                        handle,
                        mov,
                        parent: ph,
                        child: None,
                        maximum_valid_depth: self.mcts.max_depth(),
                    };
                    self.moves.push(new_move);
                    self.nodes[ph].outgoing.push(handle);
                }
                println!(
                    "      parent now has {} outgoing moves",
                    self.nodes[ph].outgoing.len()
                );
                vec![]
            }
            // We only have children.
            (true, false) => {
                // Reuse the old move by giving it the first child.
                println!(
                    "      parent has {} outgoing moves",
                    self.nodes[ph].outgoing.len()
                );
                let mut handles = vec![];
                while !child_states.is_empty() {
                    println!("        recycling one");
                    let first_state = child_states.swap_remove(0);
                    let first_move = child_moves.swap_remove(0);
                    if let Some(handle) = self.make_first_child(first_state, first_move, mov) {
                        handles.push(handle);
                        break;
                    }
                }
                if handles.is_empty() {
                    println!("        no moves. Something's wrong. :-(");
                    vec![]
                } else {
                    // Process the remaining children.
                    println!("        processing the remaining {}", child_states.len());
                    let mut rest = child_states
                        .into_iter()
                        .zip(child_moves)
                        .filter_map(|(s, m)| self.make_child(s, m, self.moves[mov].parent))
                        .collect();
                    handles.append(&mut rest);
                    // Don't push the first, since it's recycled.
                    self.nodes[ph].outgoing.extend_from_slice(&handles[1..]);
                    handles
                }
            }
            // We have children and new moves.
            (true, true) => {
                // Recycle the old move with a new move.
                println!(
                    "      parent has {} outgoing moves",
                    self.nodes[ph].outgoing.len()
                );
                println!("        recycling 1 new move");
                let first_move = new_moves.swap_remove(0);
                self.moves[mov].mov = first_move;
                // Process the remaining new moves.
                println!("        processing the remaining {}", new_moves.len());
                for mov in new_moves {
                    let handle = self.moves.len();
                    let new_move = MoveInfo {
                        handle,
                        mov,
                        parent: ph,
                        child: None,
                        maximum_valid_depth: self.mcts.max_depth(),
                    };
                    self.moves.push(new_move);
                    self.nodes[ph].outgoing.push(handle);
                }
                // Process the children.
                println!("        processing {} children", child_states.len());
                let handles = child_states
                    .into_iter()
                    .zip(child_moves)
                    .filter_map(|(s, m)| self.make_child(s, m, self.moves[mov].parent))
                    .collect::<Vec<_>>();
                self.nodes[ph].outgoing.extend_from_slice(&handles);
                println!(
                    "      parent now has {} outgoing moves",
                    self.nodes[ph].outgoing.len()
                );
                handles
            }
        }
    }
    fn make_first_child(&mut self, s: <M>::State, m: Move<M>, h: MoveHandle) -> Option<MoveHandle> {
        // Update the move.
        let node = self.find_or_make_node(s);
        // Eliminate cycles. Don't add a move if the child is the ancestor of the parent.
        if self.ancestors(self.moves[h].parent).contains(&node) || self.moves[h].parent == node {
            println!("          Cycle detected! Ignoring this child.");
            None
        } else {
            if !self.nodes[node].incoming.contains(&h) {
                self.nodes[node].incoming.push(h);
            }
            self.moves[h].child = Some(node);
            self.moves[h].mov = m;
            // NOTE: The maximum_valid_depth is examined during the update.
            Some(h)
        }
    }
    // For each state `s`, add the appropriate moves to the graph.
    fn make_child(&mut self, s: <M>::State, m: Move<M>, parent: NodeHandle) -> Option<MoveHandle> {
        let node = self.find_or_make_node(s);
        // Eliminate cycles. Don't add a move if the child is the ancestor of the parent.
        if self.ancestors(parent).contains(&node) || parent == node {
            println!(
                "          Cycle detected! Ignoring this child ({} {}).",
                self.ancestors(parent).contains(&node),
                parent == node
            );
            None
        } else {
            let handle = self.moves.len();
            if !self.nodes[node].incoming.contains(&handle) {
                self.nodes[node].incoming.push(handle);
            }
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
            println!("          created a new child!");
            Some(handle)
        }
    }
    fn find_or_make_node(&mut self, s: <M>::State) -> NodeHandle {
        match self.nodes.iter().position(|n| n.state == s) {
            Some(h) => {
                println!("          found node {}", h);
                h
            }
            None => {
                let node_handle = self.nodes.len();
                println!("          made node {}", node_handle);
                let evaluation = self.state_eval.evaluate(&s, &self.mcts);
                let q: f64 = evaluation.clone().into();
                let incoming = vec![];
                let mut outgoing = vec![];
                for m in s.available_moves(&self.mcts) {
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
                    state: s,
                    n: 1.0,
                    q,
                    evaluation,
                    incoming,
                    outgoing,
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
            self.nodes[handle].q = self.mcts.combine_qs(self.nodes[handle].q, evaluation);
            self.nodes[handle].n += 1.0;
            // TODO: this isn't quite right. What about other parents of these nodes?
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

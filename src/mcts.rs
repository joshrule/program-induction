//! Representations capable of [Monte Carlo Tree Search].
//!
//! [Monte Carlo Tree Search]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

// TODO:
// - Replace Eq constraint on State with some sort of hash for speed.
// - ask for moves & states to be Debug for better debugging.
// - Create a recycling pool for moves.
use rand::Rng;

type Move<M> = <<M as MCTS>::State as State<M>>::Move;
type StateEvaluation<M> = <<M as MCTS>::StateEval as StateEvaluator<M>>::StateEvaluation;
pub type NodeHandle = usize;
pub type MoveHandle = usize;

pub trait State<M: MCTS<State = Self>>: PartialEq + Sized + Sync {
    type Move: Clone + Sync + Send;
    type MoveList: IntoIterator<Item = Self::Move>;
    fn available_moves(&self, mcts: &mut M) -> Self::MoveList;
    fn make_move<R: Rng>(&self, mov: &Self::Move, mcts: &mut M, rng: &mut R) -> Self;
    fn add_moves_for_new_data(&self, moves: &[Self::Move], mcts: &mut M) -> Self::MoveList;
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
        let mut result = Some(0);
        while result.is_some() && !predicate(&self.tree.mcts) {
            result = self.tree.step(rng);
            steps += 1;
        }
        println!("steps: {}", steps);
    }
    // Take a single search step.
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> bool {
        self.tree.step(rng).is_some()
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
            nodes: vec![root_node],
        }
    }
    pub fn summary(&self) -> (usize, usize) {
        (self.nodes.len(), self.moves.len())
    }
    pub fn show(&self) {
        println!("SearchTree");
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
    pub fn reevaluate_states<R: Rng>(&mut self, rng: &mut R) {
        println!("#     reevaluating states");
        // iterate over the nodes and copy the evaluation information from the objects.
        println!("#       reevaluating nodes");
        for node in self.nodes.iter_mut() {
            node.evaluation = self.state_eval.evaluate(&node.state, &mut self.mcts, rng);
            println!(
                "#         new evaluation: {:.4}",
                node.evaluation.clone().into()
            );
        }
        println!("#       updating qs");
        self.recompute_qs();
    }
    pub fn recompute_qs(&mut self) {
        let mut stack = vec![self.root];
        let mut finished = vec![];
        while let Some(&node) = stack.last() {
            // println!("#      stack: {:?}", stack);
            // println!("#      finished: {:?}", finished);
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
                    maximum_valid_depth: self.mcts.max_depth(),
                };
                self.moves.push(new_move);
                self.nodes[nh].outgoing.push(handle);
            }
        }
    }
    /// Take a single search step.
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> Option<usize> {
        self.expand(rng).map(|mh| {
            // INVARIANT: Moves returned by expand always have children.
            self.backpropagate(self.moves[mh].child.unwrap());
            self.update_maximum_valid_depths(mh);
            1
        })
    }
    // Expand the search tree by taking a single move.
    fn expand<R: Rng>(&mut self, rng: &mut R) -> Option<MoveHandle> {
        if self.nodes.len() >= self.mcts.max_states() {
            println!("STOP: {} >= {}", self.nodes.len(), self.mcts.max_states());
            return None;
        }
        let mut depth = 0;
        let mut nh = self.root;
        loop {
            // Choose a move.
            let moves = self.nodes[nh].outgoing.iter().map(|n| &self.moves[*n]);
            println!("#   got some moves");
            // Looks weird, but helps prevent a borrowing issue;
            let mh = self.move_eval.choose(moves, nh, &self)?.handle;
            let mov = &self.moves[mh];
            match mov.child {
                // Descend known moves.
                Some(child_nh) => {
                    nh = child_nh;
                    depth += 1;
                    println!("#   Child is {}. Descending to depth {}.", child_nh, depth);
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
    }
    // Add `MoveInfo`s representing each of the newly created `child_states`.
    fn update_tree<R: Rng>(
        &mut self,
        child_state: <M>::State,
        mov: MoveHandle,
        rng: &mut R,
    ) -> Option<MoveHandle> {
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
            // Remove the move from the tree.
            self.nodes[ph].outgoing = self.nodes[ph]
                .outgoing
                .iter()
                .filter(|&mh| *mh == mov)
                .copied()
                .collect();
            None
        } else {
            // Give the move a child.
            self.moves[mov].child = Some(ch);
            // Add the move as a parent of the child, if new.
            if !self.nodes[ch].incoming.contains(&mov) {
                self.nodes[ch].incoming.push(mov);
            }
            Some(mov)
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
                let q: f64 = evaluation.clone().into();
                let incoming = vec![];
                let mut outgoing = vec![];
                for mov in s.available_moves(&mut self.mcts) {
                    let handle = self.moves.len();
                    let new_move = MoveInfo {
                        handle,
                        mov,
                        parent: node_handle,
                        child: None,
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
    // Bubble search statistics back to the root.
    fn backpropagate(&mut self, new_node: NodeHandle) {
        let evaluation = self.nodes[new_node].evaluation.clone().into();
        let mut stack = vec![new_node];
        while let Some(nh) = stack.pop() {
            self.nodes[nh].q = self.mcts.combine_qs(self.nodes[nh].q, evaluation);
            self.nodes[nh].n += 1.0;
            for &mh in &self.nodes[nh].incoming {
                stack.push(self.moves[mh].parent)
            }
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

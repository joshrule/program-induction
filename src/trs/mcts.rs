// use super::{TRSMoveName, TRSMoves, TRS};
// use itertools::Itertools;
// use mcts::{MoveEvaluator, State, StateEvaluator, MCTS};
use mcts::State;
use trs::{as_result, TRSMove, TRS};

pub struct TRSMCTS;

#[derive(Debug, Clone, PartialEq)]
pub struct TRSMCTSState<'a, 'b, 'c> {
    parents: Vec<&'c TRS<'a, 'b>>,
    moves: &'c [TRSMove],
}

impl<'a, 'b, 'c> State for TRSMCTSState<'a, 'b, 'c> {
    type Move = TRSMove;
    type MoveList = Vec<Self::Move>;
    fn available_moves(&self) -> Self::MoveList {
        self.moves
            .iter()
            .filter(|m| m.num_parents() == self.parents.len())
            .copied()
            .collect()
    }
    fn make_move(&self, mov: &Self::Move) -> Vec<Self> {
        // TODO: actually write this
        as_result(mv.take(&self, task, obs, rng, &self.parents, params, gpparams))
    }
    fn uniquify(taken: &Self::Move, generated: usize) -> (Vec<Self::Move>, Vec<Self::Move>) {
        // TODO: actually write this
        (vec![], vec![])
    }
}

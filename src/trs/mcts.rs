// TODO:
// - Update posterior computation to be more efficient. Let hypothesis store
//   likelihoods so just incrementally update.
// - refactor playout
// - penalize likelihood for non-list-literals
// - non-terminals collect multiple playouts
use itertools::Itertools;
use mcts::{MoveEvaluator, MoveInfo, NodeHandle, SearchTree, State, StateEvaluator, MCTS};
use polytype::TypeSchema;
use rand::{
    distributions::{Distribution, WeightedIndex},
    prelude::{IteratorRandom, SliceRandom},
    Rng,
};
use std::{collections::HashMap, convert::TryFrom};
use term_rewriting::{Atom, Context, Rule, RuleContext};
use trs::{lexicon::Environment, Composition, Hypothesis, Lexicon, ModelParams, Recursion, TRS};
use utils::logsumexp;

type RevisionHandle = usize;
type TerminalHandle = usize;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum StateHandle {
    Revision(RevisionHandle),
    Terminal(TerminalHandle),
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct MCTSState {
    handle: StateHandle,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Revise<'a, 'b> {
    n: usize,
    trs: TRS<'a, 'b>,
    spec: Option<MCTSMoveState>,
    playout: Option<TerminalHandle>,
}

pub enum StateKind<'a, 'b> {
    Terminal(Hypothesis<'a, 'b>),
    Revision(Revise<'a, 'b>),
}

pub struct TRSMCTS<'a, 'b> {
    pub lexicon: Lexicon<'b>,
    pub bg: &'a [Rule],
    pub deterministic: bool,
    pub data: &'a [Rule],
    pub revisions: Vec<Revise<'a, 'b>>,
    pub terminals: Vec<Hypothesis<'a, 'b>>,
    pub model: ModelParams,
    pub params: MCTSParams,
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct MCTSParams {
    pub max_depth: usize,
    pub max_states: usize,
    pub max_revisions: usize,
    pub max_size: usize,
    pub atom_weights: (f64, f64, f64, f64),
    pub invent: bool,
}

pub struct MCTSMoveEvaluator;

pub struct MCTSStateEvaluator;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum MCTSMoveState {
    Compose,
    Recurse,
    Variablize,
    MemorizeData(Option<usize>),
    SampleRule(RuleContext),
    RegenerateRule(Option<(usize, RuleContext)>),
    DeleteRules(Option<usize>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MCTSMove {
    MemorizeData,
    MemorizeDatum(Option<usize>),
    SampleRule,
    SampleAtom(Atom),
    RegenerateRule,
    RegenerateThisRule(usize, RuleContext),
    DeleteRules,
    DeleteRule(Option<usize>),
    Generalize,
    AntiUnify,
    Variablize(Option<(usize, Rule)>),
    Compose(Option<Composition>),
    Recurse(Option<Recursion>),
    Stop,
}

impl<'a, 'b> StateKind<'a, 'b> {
    pub fn new(trs: TRS<'a, 'b>, n: usize, mcts: &TRSMCTS<'a, 'b>) -> Self {
        if n + 1 >= mcts.params.max_revisions {
            StateKind::Terminal(Hypothesis::new(trs, &mcts.data, 1.0, mcts.model))
        } else {
            StateKind::Revision(Revise::new(trs, None, n + 1))
        }
    }
}

impl<'a, 'b> Revise<'a, 'b> {
    pub fn new(trs: TRS<'a, 'b>, spec: Option<MCTSMoveState>, n: usize) -> Self {
        Revise {
            trs,
            spec,
            n,
            playout: None,
        }
    }
    pub fn available_moves(&self, mcts: &TRSMCTS, moves: &mut Vec<MCTSMove>, _rh: RevisionHandle) {
        match self.spec {
            None => {
                // Search can always stop or sample a new rule.
                moves.push(MCTSMove::SampleRule);
                moves.push(MCTSMove::Stop);
                // A TRS must have a rule in order to regenerate or generalize.
                if !self.trs.is_empty() {
                    moves.push(MCTSMove::RegenerateRule);
                    moves.push(MCTSMove::Generalize);
                    moves.push(MCTSMove::Compose(None));
                    moves.push(MCTSMove::Recurse(None));
                    moves.push(MCTSMove::Variablize(None));
                }
                // A TRS must have >1 rule to delete without creating cycles.
                // Anti-unification relies on having two rules to unify.
                if self.trs.len() > 1 {
                    moves.push(MCTSMove::DeleteRules);
                    moves.push(MCTSMove::AntiUnify);
                }
                // We can only add data if there's data to add.
                if !mcts.data.is_empty() {
                    moves.push(MCTSMove::MemorizeData);
                }
            }
            Some(MCTSMoveState::Variablize) => {
                let ruless = self.trs.try_all_variablizations();
                for (m, rules) in ruless.into_iter().enumerate() {
                    for rule in rules {
                        moves.push(MCTSMove::Variablize(Some((m, rule))))
                    }
                }
            }
            Some(MCTSMoveState::Compose) => self
                .trs
                .find_all_compositions()
                .into_iter()
                .for_each(|composition| moves.push(MCTSMove::Compose(Some(composition)))),
            Some(MCTSMoveState::Recurse) => self
                .trs
                .find_all_recursions()
                .into_iter()
                .for_each(|recursion| moves.push(MCTSMove::Recurse(Some(recursion)))),
            Some(MCTSMoveState::MemorizeData(n)) => {
                let lower_bound = n.unwrap_or(0);
                (lower_bound..mcts.data.len())
                    .map(|i_datum| MCTSMove::MemorizeDatum(Some(i_datum)))
                    .for_each(|mv| moves.push(mv));
                if n.is_some() {
                    moves.push(MCTSMove::MemorizeDatum(None));
                }
            }
            Some(MCTSMoveState::DeleteRules(n)) => {
                let lower_bound = n.unwrap_or(0);
                (lower_bound..self.trs.len())
                    .map(|rule| MCTSMove::DeleteRule(Some(rule)))
                    .for_each(|mv| moves.push(mv));
                if n.is_some() {
                    moves.push(MCTSMove::DeleteRule(None));
                }
            }
            Some(MCTSMoveState::SampleRule(ref context))
            | Some(MCTSMoveState::RegenerateRule(Some((_, ref context)))) => {
                if let Some(place) = context.leftmost_hole() {
                    self.trs
                        .lex
                        .rulecontext_fillers(&context, &place)
                        .into_iter()
                        .map(MCTSMove::SampleAtom)
                        .for_each(|mv| moves.push(mv))
                }
            }
            Some(MCTSMoveState::RegenerateRule(None)) => {
                for (i, rule) in self.trs.utrs.rules.iter().enumerate() {
                    let rulecontext = RuleContext::from(rule.clone());
                    for (_, place) in rulecontext.subcontexts() {
                        let context = rulecontext.replace(&place, Context::Hole).unwrap();
                        moves.push(MCTSMove::RegenerateThisRule(i, context));
                    }
                }
            }
        }
    }
    pub fn make_move(
        &self,
        mv: &MCTSMove,
        mcts: &mut TRSMCTS<'a, 'b>,
    ) -> Option<StateKind<'a, 'b>> {
        match *mv {
            MCTSMove::Stop => {
                let hypothesis = Hypothesis::new(self.trs.clone(), &mcts.data, 1.0, mcts.model);
                Some(StateKind::Terminal(hypothesis))
            }
            MCTSMove::Generalize => {
                let mut trss = self.trs.generalize().ok()?;
                Some(StateKind::new(trss.swap_remove(0), self.n, mcts))
            }
            MCTSMove::AntiUnify => {
                let trs = self.trs.lgg().ok()?;
                Some(StateKind::new(trs, self.n, mcts))
            }
            MCTSMove::Compose(None) => {
                let spec = Some(MCTSMoveState::Compose);
                let state = Revise::new(self.trs.clone(), spec, self.n);
                Some(StateKind::Revision(state))
            }
            MCTSMove::Compose(Some(ref composition)) => {
                let trs = self.trs.compose_by(composition)?;
                Some(StateKind::new(trs, self.n, mcts))
            }
            MCTSMove::Recurse(None) => {
                let spec = Some(MCTSMoveState::Recurse);
                let state = Revise::new(self.trs.clone(), spec, self.n);
                Some(StateKind::Revision(state))
            }
            MCTSMove::Recurse(Some(ref recursion)) => {
                let trs = self.trs.recurse_by(recursion)?;
                Some(StateKind::new(trs, self.n, mcts))
            }
            MCTSMove::Variablize(None) => {
                let spec = Some(MCTSMoveState::Variablize);
                let state = Revise::new(self.trs.clone(), spec, self.n);
                Some(StateKind::Revision(state))
            }
            MCTSMove::Variablize(Some((m, ref rule))) => {
                let mut trs = self.trs.clone();
                trs.utrs.rules[m] = rule.clone();
                Some(StateKind::new(trs, self.n, mcts))
            }
            MCTSMove::DeleteRules => {
                // We're stating an intention: just the internal state changes.
                let state = Revise::new(
                    self.trs.clone(),
                    Some(MCTSMoveState::DeleteRules(None)),
                    self.n,
                );
                Some(StateKind::Revision(state))
            }
            MCTSMove::DeleteRule(None) => {
                // You're done deleting: the rules don't change, but internal state resets.
                Some(StateKind::new(self.trs.clone(), self.n, mcts))
            }
            MCTSMove::DeleteRule(Some(n)) => {
                // You're actively deleting or finished not by choice: rules and state change.
                let mut trs = self.trs.clone();
                trs.utrs.remove_idx(n).ok()?;
                if n >= trs.len() {
                    Some(StateKind::new(trs, self.n, mcts))
                } else {
                    let spec = Some(MCTSMoveState::DeleteRules(Some(n + 1)));
                    let state = Revise::new(trs, spec, self.n);
                    Some(StateKind::Revision(state))
                }
            }
            MCTSMove::MemorizeData => {
                let state = Revise::new(
                    self.trs.clone(),
                    Some(MCTSMoveState::MemorizeData(None)),
                    self.n,
                );
                Some(StateKind::Revision(state))
            }
            MCTSMove::MemorizeDatum(None) => {
                // You're done memorizing: the rules don't change, but internal state does.
                Some(StateKind::new(self.trs.clone(), self.n, mcts))
            }
            MCTSMove::MemorizeDatum(Some(n)) => {
                // You're actively memorizing or finished not by choice: rules and state change.
                let mut trs = self.trs.clone();
                trs.append_clauses(vec![mcts.data[n].clone()]).ok()?;
                if n + 1 == mcts.data.len() {
                    Some(StateKind::new(trs, self.n, mcts))
                } else {
                    let spec = Some(MCTSMoveState::MemorizeData(Some(n + 1)));
                    let state = Revise::new(trs, spec, self.n);
                    Some(StateKind::Revision(state))
                }
            }
            MCTSMove::SampleRule => {
                // You're stating the intention to sample: state changes.
                let spec = Some(MCTSMoveState::SampleRule(RuleContext::default()));
                let state = Revise::new(self.trs.clone(), spec, self.n);
                Some(StateKind::Revision(state))
            }
            MCTSMove::RegenerateRule => {
                // You're stating the intention to regenerate: state changes.
                let spec = Some(MCTSMoveState::RegenerateRule(None));
                let state = Revise::new(self.trs.clone(), spec, self.n);
                Some(StateKind::Revision(state))
            }
            MCTSMove::RegenerateThisRule(n, ref context) => {
                // You're stating where you want to regenerate: state changes.
                let spec = Some(MCTSMoveState::RegenerateRule(Some((n, context.clone()))));
                let state = Revise::new(self.trs.clone(), spec, self.n);
                Some(StateKind::Revision(state))
            }
            MCTSMove::SampleAtom(atom) => match self.spec {
                Some(MCTSMoveState::SampleRule(ref context)) => {
                    let place = context.leftmost_hole()?;
                    let mut trs = self.trs.clone();
                    let new_context = context.replace(&place, Context::from(atom))?;
                    if let Ok(rule) = Rule::try_from(&new_context) {
                        trs.append_clauses(vec![rule]).ok()?;
                        println!("#   \"{}\"", trs.to_string().lines().join(" "));
                        Some(StateKind::new(trs, self.n, mcts))
                    } else {
                        let spec = Some(MCTSMoveState::SampleRule(new_context));
                        let state = Revise::new(trs, spec, self.n);
                        Some(StateKind::Revision(state))
                    }
                }
                Some(MCTSMoveState::RegenerateRule(Some((n, ref context)))) => {
                    let place = context.leftmost_hole()?;
                    let mut trs = self.trs.clone();
                    let new_context = context.replace(&place, Context::from(atom))?;
                    if let Ok(rule) = Rule::try_from(&new_context) {
                        trs.utrs.remove_idx(n).ok()?;
                        trs.utrs.insert_idx(n, rule).ok()?;
                        println!("#   \"{}\"", trs.to_string().lines().join(" "));
                        Some(StateKind::new(trs, self.n, mcts))
                    } else {
                        let spec = Some(MCTSMoveState::RegenerateRule(Some((n, new_context))));
                        let state = Revise::new(trs, spec, self.n);
                        Some(StateKind::Revision(state))
                    }
                }
                _ => panic!("MCTSMoveState doesn't match MCTSMove"),
            },
        }
    }
    pub fn playout<R: Rng>(&self, mcts: &mut TRSMCTS<'a, 'b>, rng: &mut R) -> TRS<'a, 'b> {
        let mut trs = self.trs.clone();
        let mut n_revisions = self.n;
        // Finish whatever move is in progress.
        match &self.spec {
            None => (),
            Some(MCTSMoveState::Compose) => {
                let composition = trs.find_all_compositions().into_iter().choose(rng).unwrap();
                trs = trs.compose_by(&composition).unwrap();
                n_revisions += 1;
            }
            Some(MCTSMoveState::Recurse) => {
                let recursion = trs.find_all_recursions().into_iter().choose(rng).unwrap();
                trs = trs.recurse_by(&recursion).unwrap();
                n_revisions += 1;
            }
            Some(MCTSMoveState::DeleteRules(progress)) => {
                println!("#        finishing deletion");
                for idx in (progress.unwrap_or(0)..trs.len()).rev() {
                    if rng.gen() {
                        trs.utrs.remove_idx(idx).ok();
                    }
                }
                n_revisions += 1;
            }
            Some(MCTSMoveState::Variablize) => {
                let ruless = self.trs.try_all_variablizations();
                if let Some(m) = (0..ruless.len()).choose(rng) {
                    if let Some(n) = (0..ruless[m].len()).choose(rng) {
                        trs.utrs.rules[m] = ruless[m][n].clone();
                        n_revisions += 1;
                    }
                }
            }
            Some(MCTSMoveState::MemorizeData(progress)) => {
                println!("#         finishing memorization");
                let lower_bound = progress.unwrap_or(0);
                for rule in mcts.data.iter().skip(lower_bound) {
                    if rng.gen() {
                        trs.append_clauses(vec![rule.clone()]).ok();
                    }
                }
                n_revisions += 1;
            }
            Some(MCTSMoveState::RegenerateRule(progress)) => {
                let (n, context) = progress.clone().unwrap_or_else(|| {
                    let idx = (0..trs.len()).choose(rng).unwrap();
                    let rulecontext = RuleContext::from(trs.utrs.rules[idx].clone());
                    let (_, place) = rulecontext.subcontexts().into_iter().choose(rng).unwrap();
                    let context = rulecontext.replace(&place, Context::Hole).unwrap();
                    (idx, context)
                });
                println!(
                    "#         finishing regeneration with: {}",
                    context.pretty(&trs.lex.signature())
                );
                loop {
                    println!("#           looping");
                    let mut ctx = trs.lex.0.ctx.clone();
                    let mut types = HashMap::new();
                    trs.lex
                        .infer_rulecontext(&context, &mut types, &mut ctx)
                        .ok();
                    let env = Environment::from_rulecontext(&context, &types, mcts.params.invent);
                    if let Ok(rule) = trs.lex.sample_rule_from_context(
                        &context,
                        mcts.params.atom_weights,
                        mcts.params.max_size,
                        &env,
                        &mut ctx,
                        rng,
                    ) {
                        println!("#           sampled: {}", rule.pretty(&trs.lex.signature()));
                        trs.append_clauses(vec![rule]).ok();
                        trs.utrs.rules.swap_remove(n);
                        break;
                    }
                }
                n_revisions += 1;
            }
            Some(MCTSMoveState::SampleRule(ref context)) => {
                println!(
                    "#         finalizing sample: {}",
                    context.pretty(&trs.lex.signature())
                );
                loop {
                    println!("#           looping");
                    let mut ctx = trs.lex.0.ctx.clone();
                    let mut types = HashMap::new();
                    trs.lex
                        .infer_rulecontext(&context, &mut types, &mut ctx)
                        .ok();
                    let env = Environment::from_rulecontext(&context, &types, mcts.params.invent);
                    if let Ok(rule) = trs.lex.sample_rule_from_context(
                        context,
                        mcts.params.atom_weights,
                        mcts.params.max_size,
                        &env,
                        &mut ctx,
                        rng,
                    ) {
                        println!("#           sampled: {}", rule.pretty(&trs.lex.signature()));
                        trs.append_clauses(vec![rule]).ok();
                        break;
                    }
                }
                n_revisions += 1;
            }
        }
        // Run the remaining moves
        while n_revisions < mcts.params.max_revisions {
            let moves = WeightedIndex::new(&[
                // Memorize Data
                (!mcts.data.is_empty() as usize as f64),
                // Sample Rule
                1.0,
                // Regenerate
                (!trs.is_empty() as usize as f64),
                // Delete
                ((trs.len() > 1) as usize as f64),
                // Variablize
                (!trs.is_empty() as usize as f64),
                // Generalize
                (!trs.is_empty() as usize as f64),
                // Compose
                (!trs.is_empty() as usize as f64),
                // Recurse
                (!trs.is_empty() as usize as f64),
                // AntiUnify
                ((trs.len() > 1) as usize as f64),
                // Stop
                1.0,
            ])
            .unwrap();
            match moves.sample(rng) {
                0 => {
                    println!("#         adding data");
                    for rule in mcts.data {
                        if rng.gen() {
                            trs.append_clauses(vec![rule.clone()]).ok();
                        }
                    }
                    n_revisions += 1;
                }
                1 => {
                    let schema = TypeSchema::Monotype(trs.lex.0.to_mut().ctx.new_variable());
                    println!("#         sampling a rule");
                    loop {
                        println!("#           looping");
                        let mut ctx = trs.lex.0.ctx.clone();
                        if let Ok(rule) = trs.lex.sample_rule(
                            &schema,
                            mcts.params.atom_weights,
                            mcts.params.max_size,
                            mcts.params.invent,
                            &mut ctx,
                            rng,
                        ) {
                            println!("#           sampled: {}", rule.pretty(&trs.lex.signature()));
                            trs.append_clauses(vec![rule]).ok();
                            break;
                        }
                    }
                    n_revisions += 1;
                }
                2 => {
                    let idx = (0..trs.len()).choose(rng).unwrap();
                    let rulecontext = RuleContext::from(trs.utrs.rules[idx].clone());
                    let (_, place) = rulecontext.subcontexts().into_iter().choose(rng).unwrap();
                    let context = rulecontext.replace(&place, Context::Hole).unwrap();
                    println!(
                        "#         regenerating: {}",
                        context.pretty(&trs.lex.signature())
                    );
                    loop {
                        println!("#           looping");
                        let mut ctx = trs.lex.0.ctx.clone();
                        let mut types = HashMap::new();
                        trs.lex
                            .infer_rulecontext(&context, &mut types, &mut ctx)
                            .ok();
                        let env =
                            Environment::from_rulecontext(&context, &types, mcts.params.invent);
                        if let Ok(rule) = trs.lex.sample_rule_from_context(
                            &context,
                            mcts.params.atom_weights,
                            mcts.params.max_size,
                            &env,
                            &mut ctx,
                            rng,
                        ) {
                            println!("#           sampled: {}", rule.pretty(&trs.lex.signature()));
                            trs.utrs.remove_idx(idx).ok();
                            trs.utrs.insert_idx(idx, rule).ok();
                            break;
                        }
                    }
                    n_revisions += 1;
                }
                3 => {
                    println!("#         deleting rules");
                    for idx in (0..trs.len()).rev() {
                        if rng.gen() {
                            trs.utrs.remove_idx(idx).ok();
                        }
                    }
                    n_revisions += 1;
                }
                4 => {
                    println!("#         variablizing");
                    let (rules, combos) = trs.analyze_variablizations();
                    if let Some(combo) = combos.choose(rng) {
                        for (i, &idx) in combo.iter().enumerate() {
                            trs.utrs.rules[idx] = rules[i][idx].clone();
                        }
                        n_revisions += 1;
                    }
                }
                5 => {
                    println!("#         generalizing");
                    if let Ok(mut trss) = self.trs.generalize() {
                        trs = trss.swap_remove(0);
                        n_revisions += 1;
                    }
                }
                6 => {
                    println!("#         composing");
                    if let Some(new_trs) = trs
                        .find_all_compositions()
                        .into_iter()
                        .choose(rng)
                        .and_then(|composition| trs.compose_by(&composition))
                    {
                        trs = new_trs;
                        n_revisions += 1;
                    }
                }
                7 => {
                    println!("#         recursing");
                    if let Some(new_trs) = trs
                        .find_all_recursions()
                        .into_iter()
                        .choose(rng)
                        .and_then(|recursion| trs.recurse_by(&recursion))
                    {
                        trs = new_trs;
                        n_revisions += 1;
                    }
                }
                8 => {
                    println!("#         anti-unifying");
                    if let Ok(new_trs) = trs.lgg() {
                        trs = new_trs;
                        n_revisions += 1;
                    }
                }
                9 => {
                    println!("#         stopping");
                    break;
                }
                _ => unreachable!(),
            }
        }
        trs
    }
}

impl<'a, 'b> State<TRSMCTS<'a, 'b>> for MCTSState {
    type Move = MCTSMove;
    type MoveList = Vec<Self::Move>;
    fn available_moves(&self, mcts: &mut TRSMCTS) -> Self::MoveList {
        let mut moves = vec![];
        match self.handle {
            StateHandle::Terminal(..) => (),
            StateHandle::Revision(rh) => mcts.revisions[rh].available_moves(mcts, &mut moves, rh),
        }
        moves
    }
    fn make_move<R: Rng>(
        &self,
        mv: &Self::Move,
        mcts: &mut TRSMCTS<'a, 'b>,
        _rng: &mut R,
    ) -> Option<Self> {
        let state = match self.handle {
            StateHandle::Terminal(..) => panic!("inconsistent state: no move from terminal"),
            StateHandle::Revision(rh) => mcts.make_move(mv, rh),
            //StateHandle::Revision(rh) => mcts.revisions[rh].make_move(mv, mcts),
        }?;
        Some(mcts.add_state(state))
    }
    fn add_moves_for_new_data(&self, moves: &[Self::Move], mcts: &mut TRSMCTS) -> Vec<Self::Move> {
        self.available_moves(mcts)
            .into_iter()
            .filter(|m| !moves.contains(&m))
            .collect()
    }
}

impl<'a, 'b> MCTS for TRSMCTS<'a, 'b> {
    type StateEval = MCTSStateEvaluator;
    type MoveEval = MCTSMoveEvaluator;
    type State = MCTSState;
    fn max_depth(&self) -> usize {
        self.params.max_depth
    }
    fn max_states(&self) -> usize {
        self.params.max_states
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
        model: ModelParams,
        params: MCTSParams,
    ) -> TRSMCTS<'a, 'b> {
        TRSMCTS {
            lexicon,
            bg,
            deterministic,
            data,
            model,
            params,
            terminals: vec![],
            revisions: vec![],
        }
    }
    fn make_move(&mut self, mv: &MCTSMove, rh: RevisionHandle) -> Option<StateKind<'a, 'b>> {
        // TODO: remove clone, perhaps by pull revision make move up to this level?
        let revision = self.revisions[rh].clone();
        revision.make_move(mv, self)
    }
    pub fn add_state(&mut self, state: StateKind<'a, 'b>) -> MCTSState {
        match state {
            StateKind::Terminal(h) => self.add_terminal(h),
            StateKind::Revision(r) => self.add_revision(r),
        }
    }
    pub fn add_revision(&mut self, state: Revise<'a, 'b>) -> MCTSState {
        self.revisions.push(state);
        let handle = StateHandle::Revision(self.revisions.len() - 1);
        MCTSState { handle }
    }
    pub fn add_terminal(&mut self, state: Hypothesis<'a, 'b>) -> MCTSState {
        self.terminals.push(state);
        let handle = StateHandle::Terminal(self.terminals.len() - 1);
        MCTSState { handle }
    }
    pub fn root(&mut self) -> MCTSState {
        let state = Revise {
            trs: TRS::new_unchecked(&self.lexicon, self.deterministic, self.bg, vec![]),
            spec: None,
            n: 0,
            playout: None,
        };
        self.add_revision(state)
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
    fn zero(&self) -> f64 {
        std::f64::NEG_INFINITY
    }
    fn evaluate<R: Rng>(
        &self,
        state: &<TRSMCTS<'a, 'b> as MCTS>::State,
        mcts: &mut TRSMCTS<'a, 'b>,
        rng: &mut R,
    ) -> Self::StateEvaluation {
        println!("#     evaluating");
        match state.handle {
            StateHandle::Terminal(th) => {
                println!("#       node is terminal");
                mcts.terminals[th].lposterior
            }
            StateHandle::Revision(rh) => match &mcts.revisions[rh].playout {
                Some(th) => {
                    println!("#       found a playout");
                    mcts.terminals[*th].lposterior
                }
                None => {
                    println!("#       playing out");
                    // TODO: remove clone
                    let revision = mcts.revisions[rh].clone();
                    let trs = revision.playout(mcts, rng);
                    println!(
                        "#         simulated: \"{}\"",
                        trs.to_string().lines().join(" ")
                    );
                    let h = Hypothesis::new(trs, &mcts.data, 1.0, mcts.model);
                    let score = h.lposterior;
                    let th = mcts.terminals.len();
                    mcts.terminals.push(h);
                    mcts.revisions[rh].playout = Some(th);
                    score
                }
            },
        }
    }
}

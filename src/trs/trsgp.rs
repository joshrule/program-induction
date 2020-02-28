use super::{TRSMoveName, TRSMoves, TRS};
use gp::{GPParams, Tournament, GP};
use itertools::Itertools;
use polytype::TypeSchema;
use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng,
};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use term_rewriting::{Rule, RuleContext};
use trs::{Lexicon, ModelParams, TRSMove};
use Task;

type Parents<'a, 'b> = Vec<TRS<'a, 'b>>;
type Tried<'a, 'b> = HashMap<TRSMoveName, Vec<Parents<'a, 'b>>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Parameters for [`Lexicon`] genetic programming ([`GP`]).
///
/// [`Lexicon`]: struct.Lexicon.html
/// [`GP`]: ../trait.GP.html
pub struct GeneticParams {
    // A list of the moves available during search.
    pub moves: TRSMoves,
    /// The maximum number of nodes a sampled `Term` can have without failing.
    pub max_sample_size: usize,
    /// The weight to assign variables, constants, and non-constant operators, respectively.
    pub atom_weights: (f64, f64, f64, f64),
    /// `true` if you want only deterministic TRSs during search, else `false`.
    pub deterministic: bool,
    /// The number of times search can recurse.
    pub depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticParamsFull {
    // A list of the moves available during search.
    pub moves: TRSMoves,
    /// The maximum number of nodes a sampled `Term` can have without failing.
    pub max_sample_size: usize,
    /// The weight to assign variables, constants, and non-constant operators, respectively.
    pub atom_weights: (f64, f64, f64, f64),
    /// `true` if you want only deterministic TRSs during search, else `false`.
    pub deterministic: bool,
    /// The number of times search can recurse.
    pub depth: usize,
    /// A set of `ModelParams` to support search recursion.
    pub model: ModelParams,
}
impl GeneticParamsFull {
    pub fn new(g: &GeneticParams, m: ModelParams) -> Self {
        GeneticParamsFull {
            moves: g.moves.clone(),
            max_sample_size: g.max_sample_size,
            atom_weights: g.atom_weights,
            deterministic: g.deterministic,
            depth: g.depth,
            model: m,
        }
    }
}

pub struct TRSGP<'a, 'b> {
    pub lexicon: Lexicon<'b>,
    pub bg: &'a [Rule],
    pub contexts: Vec<RuleContext>,
    pub(crate) tried: Arc<RwLock<Tried<'a, 'b>>>,
}
    pub fn new<'c, 'd>(
        lex: &Lexicon<'d>,
        bg: &'c [Rule],
        contexts: Vec<RuleContext>,
    ) -> GPLexicon<'c, 'd> {
impl<'a, 'b> TRSGP<'a, 'b> {
        let lexicon = lex.clone();
        let tried = Arc::new(RwLock::new(HashMap::new()));
        TRSGP {
            lexicon,
            bg,
            tried,
            contexts,
        }
    }
    pub fn clear(&self) {
        let mut tried = self.tried.write().expect("poisoned");
        *tried = HashMap::new();
    }
    pub fn add(&self, name: TRSMoveName, parents: Parents<'a, 'b>) {
        let mut tried = self.tried.write().expect("poisoned");
        let entry = tried.entry(name).or_insert_with(|| vec![]);
        entry.push(parents);
    }
    pub fn check(&self, name: TRSMoveName, parents: &[&TRS]) -> bool {
        let mut tried = self.tried.write().expect("poisoned");
        let past_parents = tried.entry(name).or_insert_with(|| vec![]);
        self.novelty_possible(name, parents, past_parents)
    }
    fn novelty_possible(&self, name: TRSMoveName, parents: &[&TRS], past: &[Vec<TRS>]) -> bool {
        let check = || {
            !past
                .iter()
                .any(|p| parents.iter().zip(p).all(|(x, y)| *x != y))
        };
        match name {
            TRSMoveName::Memorize => past.is_empty(),
            TRSMoveName::Compose
            | TRSMoveName::ComposeDeep
            | TRSMoveName::RecurseDeep
            | TRSMoveName::GeneralizeDeep
            | TRSMoveName::Recurse
            | TRSMoveName::SampleRule
            | TRSMoveName::RegenerateRule => true,
            TRSMoveName::LocalDifference => parents[0].len() > 1 || check(),
            _ => check(),
        }
    }
}
impl<'a, 'b> GP for TRSGP<'a, 'b> {
    type Representation = Lexicon<'b>;
    type Expression = TRS<'a, 'b>;
    type Params = GeneticParamsFull;
    type Observation = Vec<Rule>;
    fn genesis<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        pop_size: usize,
        _tp: &TypeSchema,
    ) -> Vec<Self::Expression> {
        let trs = TRS::new(&self.lexicon, params.deterministic, self.bg, vec![]);
        match trs {
            Ok(mut trs) => {
                if params.deterministic {
                    trs.utrs.make_deterministic();
                }
                let mut pop = Vec::with_capacity(pop_size);
                while pop.len() < pop_size {
                    let sample_result = trs.sample_rule(
                        &self.contexts,
                        params.atom_weights,
                        params.max_sample_size,
                        rng,
                    );
                    if let Ok(mut new_trs) = sample_result {
                        if new_trs[0].unique_shape(&pop) {
                            pop.append(&mut new_trs);
                        }
                    }
                }
                pop
            }
            Err(err) => panic!("invalid background knowledge: {}", err),
        }
    }
    fn reproduce<R: Rng>(
        &self,
        task: &Task<Self::Representation, Self::Expression, Self::Observation>,
        rng: &mut R,
        params: &Self::Params,
        gpparams: &GPParams,
        obs: &Self::Observation,
        tournament: &Tournament<Self::Expression>,
    ) -> Vec<Self::Expression> {
        let weights = params
            .moves
            .iter()
            .map(|mv| match mv.mv {
                TRSMove::ComposeDeep => mv.weight * ((params.depth == 2) as usize),
                _ => mv.weight,
            })
            .collect_vec();
        let dist = WeightedIndex::new(weights).unwrap();
        loop {
            // Choose a move
            let choice = dist.sample(rng);
            let mv = params.moves[choice].mv;
            let name = mv.name();
            // Sample the parents.
            let parents = mv.get_parents(&tournament, rng);
            // Check the parents.
            if self.check(name, &parents) {
                // Take the move.
                // println!("#     ### {:?}", name);
                if let Ok(trss) = mv.take(&self, task, obs, rng, &parents, params, gpparams) {
                    self.add(name, parents.iter().map(|&t| t.clone()).collect());
                    // println!("#     ### {} ", trss.len());
                    return trss;
                }
            }
        }
    }
    fn validate_offspring(
        &self,
        _params: &Self::Params,
        population: &[(Self::Expression, f64)],
        _children: &[Self::Expression],
        _seen: &mut Vec<Self::Expression>,
        offspring: &mut Vec<Self::Expression>,
        max_validated: usize,
    ) {
        let mut validated = 0;
        while validated < max_validated && validated < offspring.len() {
            let x = &offspring[validated];
            let pop_unique = !population.iter().any(|p| TRS::same_shape(&p.0, &x));
            // TODO: we're removing the restriction of uniqueness to accommodate annealing.
            // With the MCTS-like idea I had recently, we'd restore something like it.
            // if pop_unique && x.unique_shape(seen) {
            if pop_unique {
                validated += 1;
            } else {
                offspring.swap_remove(validated);
            }
        }
        offspring.truncate(validated);
        // TODO: we're basically ignoring seen, see comment above
        // seen.extend_from_slice(&offspring);
    }
}

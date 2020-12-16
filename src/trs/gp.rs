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
use term_rewriting::Rule;
use trs::{Lexicon, ModelParams, SampleError, TRS};
use Task;

pub type Moves = Vec<WeightedMove>;
type Parents<'a, 'b> = Vec<TRS<'a, 'b>>;
type Tried<'a, 'b> = HashMap<MoveName, Vec<Parents<'a, 'b>>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Parameters for [`Lexicon`] genetic programming ([`GP`]).
///
/// [`Lexicon`]: struct.Lexicon.html
/// [`GP`]: ../trait.GP.html
pub struct GeneticParams {
    // A list of the moves available during search.
    pub moves: Moves,
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
    pub moves: Moves,
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

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct WeightedMove {
    pub weight: usize,
    pub mv: Move,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MoveName {
    Memorize,
    SampleRule,
    RegenerateRule,
    LocalDifference,
    MemorizeOne,
    DeleteRule,
    Variablize,
    Generalize,
    Recurse,
    DeleteRules,
    Combine,
    Compose,
    ComposeDeep,
    RecurseDeep,
    GeneralizeDeep,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Move {
    Memorize(bool),
    SampleRule((f64, f64, f64, f64), usize),
    RegenerateRule((f64, f64, f64, f64), usize),
    LocalDifference,
    MemorizeOne,
    DeleteRule,
    Variablize,
    Generalize,
    Recurse(usize),
    DeleteRules(usize),
    Combine(usize),
    Compose,
    ComposeDeep,
    RecurseDeep(usize),
    GeneralizeDeep,
}

pub struct TRSGP<'a, 'b> {
    pub lexicon: Lexicon<'b>,
    pub bg: &'a [Rule],
    pub(crate) tried: Arc<RwLock<Tried<'a, 'b>>>,
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

impl Move {
    #[allow(clippy::too_many_arguments)]
    pub fn take<'a, 'b, R: Rng>(
        &self,
        gp: &TRSGP<'a, 'b>,
        task: &Task<Lexicon<'b>, TRS<'a, 'b>, Vec<Rule>>,
        obs: &[Rule],
        rng: &mut R,
        parents: &[&TRS<'a, 'b>],
        params: &GeneticParamsFull,
        gpparams: &GPParams,
    ) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        match *self {
            Move::Memorize(deterministic) => {
                Ok(TRS::memorize(&gp.lexicon, deterministic, &gp.bg, obs))
            }
            Move::SampleRule(aw, mss) => parents[0].sample_rule(aw, mss, rng),
            Move::RegenerateRule(aw, mss) => parents[0].regenerate_rule(aw, mss, rng),
            Move::LocalDifference => parents[0].local_difference(rng),
            Move::MemorizeOne => parents[0].memorize_one(obs),
            Move::DeleteRule => parents[0].delete_rule(),
            Move::Variablize => parents[0].variablize(),
            Move::Generalize => parents[0].generalize().map(|trs| vec![trs]),
            Move::Recurse(n) => parents[0].recurse(n),
            Move::DeleteRules(t) => parents[0].delete_rules(rng, t),
            Move::Combine(t) => TRS::combine(&parents[0], &parents[1], rng, t),
            Move::Compose => parents[0].compose(),
            Move::ComposeDeep => parents[0]
                .compose()
                .and_then(|trss| TRS::nest(&trss, task, gp, rng, params, gpparams)),
            Move::RecurseDeep(n) => parents[0]
                .recurse(n)
                .and_then(|trss| TRS::nest(&trss, task, gp, rng, params, gpparams)),
            Move::GeneralizeDeep => parents[0]
                .generalize()
                .and_then(|trs| TRS::nest(&[trs], task, gp, rng, params, gpparams)),
        }
    }
    pub fn get_parents<'a, 'b, 'c, R: Rng>(
        &self,
        t: &'c Tournament<TRS<'a, 'b>>,
        rng: &mut R,
    ) -> Vec<&'c TRS<'a, 'b>> {
        match *self {
            Move::Memorize(_) => vec![],
            Move::Combine(_) => vec![t.sample(rng), t.sample(rng)],
            _ => vec![t.sample(rng)],
        }
    }
    pub(crate) fn name(&self) -> MoveName {
        match *self {
            Move::Memorize(_) => MoveName::Memorize,
            Move::SampleRule(..) => MoveName::SampleRule,
            Move::RegenerateRule(..) => MoveName::RegenerateRule,
            Move::LocalDifference => MoveName::LocalDifference,
            Move::MemorizeOne => MoveName::MemorizeOne,
            Move::DeleteRule => MoveName::DeleteRule,
            Move::Variablize => MoveName::Variablize,
            Move::Generalize => MoveName::Generalize,
            Move::Recurse(..) => MoveName::Recurse,
            Move::DeleteRules(..) => MoveName::DeleteRules,
            Move::Combine(..) => MoveName::Combine,
            Move::Compose => MoveName::Compose,
            Move::ComposeDeep => MoveName::ComposeDeep,
            Move::RecurseDeep(..) => MoveName::RecurseDeep,
            Move::GeneralizeDeep => MoveName::GeneralizeDeep,
        }
    }
}

impl<'a, 'b> TRSGP<'a, 'b> {
    pub fn new<'c, 'd>(lex: &Lexicon<'d>, bg: &'c [Rule]) -> TRSGP<'c, 'd> {
        let lexicon = lex.clone();
        let tried = Arc::new(RwLock::new(HashMap::new()));
        TRSGP { lexicon, bg, tried }
    }
    pub fn clear(&self) {
        let mut tried = self.tried.write().expect("poisoned");
        *tried = HashMap::new();
    }
    pub fn add(&self, name: MoveName, parents: Parents<'a, 'b>) {
        let mut tried = self.tried.write().expect("poisoned");
        let entry = tried.entry(name).or_insert_with(|| vec![]);
        entry.push(parents);
    }
    pub fn check(&self, name: MoveName, parents: &[&TRS]) -> bool {
        let mut tried = self.tried.write().expect("poisoned");
        let past_parents = tried.entry(name).or_insert_with(|| vec![]);
        self.novelty_possible(name, parents, past_parents)
    }
    fn novelty_possible(&self, name: MoveName, parents: &[&TRS], past: &[Vec<TRS>]) -> bool {
        let check = || {
            !past
                .iter()
                .any(|p| parents.iter().zip(p).all(|(x, y)| *x != y))
        };
        match name {
            MoveName::Memorize => past.is_empty(),
            MoveName::Compose
            | MoveName::ComposeDeep
            | MoveName::RecurseDeep
            | MoveName::GeneralizeDeep
            | MoveName::Recurse
            | MoveName::SampleRule
            | MoveName::RegenerateRule => true,
            MoveName::LocalDifference => parents[0].len() > 1 || check(),
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
                    let sample_result =
                        trs.sample_rule(params.atom_weights, params.max_sample_size, rng);
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
                Move::ComposeDeep => mv.weight * ((params.depth == 2) as usize),
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
                if let Ok(trss) = mv.take(&self, task, obs, rng, &parents, params, gpparams) {
                    self.add(name, parents.iter().map(|&t| t.clone()).collect());
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

impl<'a, 'b, 'c> From<&'c TRSGP<'a, 'b>> for &'c Lexicon<'b> {
    fn from(gp_lex: &'c TRSGP<'a, 'b>) -> &'c Lexicon<'b> {
        &gp_lex.lexicon
    }
}

impl<'ctx, 'b> TRS<'ctx, 'b> {
    pub fn nest<R: Rng>(
        trss: &[Self],
        task: &Task<Lexicon<'ctx, 'b>, Self, Vec<Rule>>,
        gp_lex: &TRSGP<'ctx, 'b>,
        rng: &mut R,
        params: &GeneticParamsFull,
        gpparams: &GPParams,
    ) -> Result<Vec<Self>, SampleError<'ctx>> {
        let new_trss = trss
            .iter()
            .filter_map(|trs| trs.recurse_search(task, gp_lex, params, gpparams, rng).ok())
            .collect_vec();
        as_result(new_trss)
    }
    fn recurse_search<R: Rng>(
        &self,
        task: &Task<Lexicon<'ctx, 'b>, Self, Vec<Rule>>,
        gp_lex: &TRSGP<'ctx, 'b>,
        genetic: &GeneticParamsFull,
        gp: &GPParams,
        rng: &mut R,
    ) -> Result<Self, SampleError<'ctx>> {
        let mut new_genetic = genetic.clone();
        new_genetic.depth = 1.max(genetic.depth) - 1;
        let score = (task.oracle)(&self.lex, self);
        let mut seen = vec![self.clone()];
        let mut pop = vec![(self.clone(), score)];
        // TODO: constant is a HACK!
        for _ in 0..20 {
            gp_lex.evolve(&new_genetic, rng, &gp, task, &mut seen, &mut pop);
        }
        Ok(pop[0].0.clone())
    }
}

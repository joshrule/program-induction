use gp::{GPParams, GP};
use itertools::Itertools;
use rand::Rng;
use term_rewriting::Rule;
use trs::{as_result, GeneticParamsFull, Lexicon, SampleError, TRS, TRSGP};
use Task;

impl<'a, 'b> TRS<'a, 'b> {
    pub fn nest<R: Rng>(
        trss: &[TRS<'a, 'b>],
        task: &Task<Lexicon<'b>, TRS<'a, 'b>, Vec<Rule>>,
        gp_lex: &TRSGP<'a, 'b>,
        rng: &mut R,
        params: &GeneticParamsFull,
        gpparams: &GPParams,
    ) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        let new_trss = trss
            .iter()
            .filter_map(|trs| trs.recurse_search(task, gp_lex, params, gpparams, rng).ok())
            .collect_vec();
        as_result(new_trss)
    }
    fn recurse_search<R: Rng>(
        &self,
        task: &Task<Lexicon<'b>, TRS<'a, 'b>, Vec<Rule>>,
        gp_lex: &TRSGP<'a, 'b>,
        genetic: &GeneticParamsFull,
        gp: &GPParams,
        rng: &mut R,
    ) -> Result<TRS<'a, 'b>, SampleError> {
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

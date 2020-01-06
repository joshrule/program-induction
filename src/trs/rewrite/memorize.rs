use super::{super::as_result, Lexicon, SampleError, TRS};
use itertools::Itertools;
use term_rewriting::Rule;

impl TRS {
    /// Given a list of `Rule`s considered to be data, memorize them all.
    pub fn memorize(lex: &Lexicon, background: Vec<Rule>, data: &[Rule]) -> Vec<TRS> {
        vec![TRS::new_unchecked(lex, background, data.to_vec())]
    }

    /// Given a list of `Rule`s considered to be data, memorize a single datum.
    pub fn memorize_one(&self, data: &[Rule]) -> Result<Vec<TRS>, SampleError> {
        let mut rules = data.to_vec();
        self.lex.filter_background(&mut rules);
        let results = rules
            .into_iter()
            .filter_map(|rule| {
                let mut trs = self.clone();
                if trs.utrs.push(rule).is_ok() {
                    Some(trs)
                } else {
                    None
                }
            })
            .collect_vec();
        as_result(results)
    }
}

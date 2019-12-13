use super::{Lexicon, SampleError, TRS};
use itertools::Itertools;
use term_rewriting::Rule;

impl TRS {
    /// Given a list of `Rule`s considered to be data, memorize them all.
    pub fn memorize(lex: &Lexicon, data: &[Rule]) -> Vec<TRS> {
        vec![TRS::new_unchecked(lex, data.to_vec())]
    }

    /// Given a list of `Rule`s considered to be data, memorize a single datum.
    pub fn memorize_one(&self, data: &[Rule]) -> Result<Vec<TRS>, SampleError> {
        let results = data
            .iter()
            .filter_map(|d| {
                let mut trs = self.clone();
                if trs.utrs.push(d.clone()).is_ok() {
                    Some(trs)
                } else {
                    None
                }
            })
            .collect_vec();
        if results.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(results)
        }
    }
}

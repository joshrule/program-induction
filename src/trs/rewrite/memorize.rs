use super::{super::as_result, Lexicon, SampleError, TRS};
use itertools::Itertools;
use term_rewriting::Rule;

impl<'a, 'b> TRS<'a, 'b> {
    /// Given a list of `Rule`s considered to be data, memorize them all.
    pub fn memorize<'c, 'd>(
        lex: &Lexicon<'d>,
        deterministic: bool,
        background: &'c [Rule],
        data: &[Rule],
    ) -> Vec<TRS<'c, 'd>> {
        vec![TRS::new_unchecked(
            lex,
            deterministic,
            background,
            data.to_vec(),
        )]
    }

    /// Given a list of `Rule`s considered to be data, memorize a single datum.
    pub fn memorize_one(&self, data: &[Rule]) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        let mut rules = data.to_vec();
        self.filter_background(&mut rules);
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

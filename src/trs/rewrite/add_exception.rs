use itertools::Itertools;
use term_rewriting::Rule;

use super::{SampleError, TRS};

impl TRS {
    /// Given a list of `Rule`s considered to be data, add one datum as a rule.
    pub fn add_exception(&self, data: &[Rule]) -> Result<Vec<TRS>, SampleError> {
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

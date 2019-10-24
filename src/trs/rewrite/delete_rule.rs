use itertools::Itertools;

use super::{SampleError, TRS};

impl TRS {
    /// Delete a rule from the rewrite system if possible. Background knowledge
    /// cannot be deleted.
    pub fn delete_rule(&self) -> Result<Vec<TRS>, SampleError> {
        let background = &self.lex.0.read().expect("poisoned lexicon").background;
        let clauses = self.utrs.clauses();
        let deletable = clauses
            .iter()
            .filter(|c| !background.contains(c))
            .collect_vec();
        if deletable.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            deletable
                .iter()
                .map(|d| {
                    let mut trs = self.clone();
                    trs.utrs.remove_clauses(d)?;
                    Ok(trs)
                })
                .collect()
        }
    }
}

use super::{super::as_result, SampleError, TRS};
use rand::Rng;
use term_rewriting::{Context, RuleContext};

impl<'a, 'b> TRS<'a, 'b> {
    /// Regenerate some portion of a rule
    pub fn regenerate_rule<R: Rng>(
        &self,
        atom_weights: (f64, f64, f64, f64),
        max_size: usize,
        rng: &mut R,
    ) -> Result<Vec<TRS<'a, 'b>>, SampleError> {
        let (n, clause) = self.choose_clause(rng)?;
        let rulecontext = RuleContext::from(clause.clone());
        let subcontexts = rulecontext.subcontexts();
        let mut trss = Vec::with_capacity(subcontexts.len());
        for (_, place) in &subcontexts {
            for _attempt in 0..100 {
                let template = rulecontext.replace(&place, Context::Hole).unwrap();
                let mut trs = self.clone();
                let sample_result = trs
                    .lex
                    .sample_rule_from_context(template, atom_weights, true, max_size, rng)
                    .drop();
                if let Ok(new_clause) = sample_result {
                    if new_clause.lhs != new_clause.rhs().unwrap() {
                        let mut new_rules = vec![new_clause];
                        trs.filter_background(&mut new_rules);
                        if !new_rules.is_empty()
                            && trs.replace(n, &clause, new_rules.pop().unwrap()).is_ok()
                        {
                            trss.push(trs);
                            break;
                        }
                    }
                }
            }
        }
        as_result(trss)
    }
}

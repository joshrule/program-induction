use itertools::Itertools;
use rand::Rng;
use std::collections::HashMap;
use term_rewriting::{Rule, Term, TRS as UntypedTRS};

use super::{SampleError, TRS};

impl TRS {
    /// Replace a subterm of the rule with a variable.
    pub fn variablize<R: Rng>(&self, rng: &mut R) -> Result<Vec<TRS>, SampleError> {
        let (n, clause) = self.choose_clause(rng)?;
        let mut types = HashMap::new();
        self.lex.infer_rule(&clause, &mut types).drop()?;
        let new_trss = clause
            .lhs
            .subterms()
            .iter()
            .unique_by(|(t, _)| t)
            .filter_map(|(t, p)| {
                let mut real_p = vec![0]; // indicates that p is from the LHS
                real_p.extend_from_slice(p);
                let mut trs = self.clone();
                let term_type = &types[&real_p];
                let new_var = trs.lex.invent_variable(term_type);
                let new_term = Term::Variable(new_var);
                let new_lhs = clause.lhs.replace_all(t, &new_term);
                let new_rhs = clause.rhs().unwrap().replace_all(t, &new_term);
                let new_clause = Rule::new(new_lhs, vec![new_rhs])?;
                trs.replace(n, &clause, new_clause).ok()?;
                if !UntypedTRS::alphas(&trs.utrs, &self.utrs) {
                    trs.smart_delete(n, n + 1).ok()
                } else {
                    None
                }
            })
            .flatten()
            .collect_vec();
        if new_trss.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(new_trss)
        }
    }
}

#[cfg(test)]
mod tests {
    use polytype::Context as TypeContext;
    use rand::thread_rng;
    use trs::parser::{parse_lexicon, parse_trs};

    #[test]
    fn variablize_test() {
        let lex = parse_lexicon(
            &[
                "+/2: INT -> INT -> INT;",
                " */2: INT -> INT -> INT;",
                " ^/2: INT -> INT -> INT;",
                " 0/0: INT; 1/0: INT; 2/0: INT;",
                " 3/0: INT; 4/0: INT; 6/0: INT;",
                " 9/0: INT;",
            ]
            .join(" "),
            "",
            "",
            true,
            TypeContext::default(),
        )
        .unwrap();
        let trs = parse_trs(
            "^(+(x_ 1) 2) = +(^(x_ 2) +(*(2 x_) 1)); ^(+(x_ 2) 2) = +(^(x_ 2) +(*(4 x_) 4)); ^(+(x_ 3) 2) = +(^(x_ 2) +(*(6 x_) 9));",
            &lex,
        )
            .expect("parsed trs");
        let mut rng = thread_rng();
        let trss = trs.variablize(&mut rng).unwrap();

        for trs in &trss {
            println!("{}\n", trs);
        }

        assert!(trss.len() == 4 || trss.len() == 5 || trss.len() == 8);
    }
}

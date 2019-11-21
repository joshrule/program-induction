use itertools::Itertools;
use std::collections::HashMap;
use term_rewriting::{Rule, Term, TRS as UntypedTRS};

use super::{SampleError, TRS};

impl TRS {
    /// Replace a subterm of the rule with a variable.
    pub fn variablize(&self) -> Result<Vec<TRS>, SampleError> {
        let new_trss = self
            .clauses()
            .into_iter()
            .flat_map(|(n, clause)| self.variablize_once(n, clause))
            .collect_vec();
        if new_trss.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            Ok(new_trss)
        }
    }
    pub fn variablize_once(&self, n: usize, clause: Rule) -> Vec<TRS> {
        let mut types = HashMap::new();
        if self.lex.infer_rule(&clause, &mut types).drop().is_err() {
            return vec![];
        }
        clause
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
            .collect_vec()
    }
}

#[cfg(test)]
mod tests {
    use polytype::Context as TypeContext;
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
        let trss = trs.variablize().unwrap();

        for trs in &trss {
            println!("{}\n", trs);
        }

        assert!(trss.len() == 4 || trss.len() == 5 || trss.len() == 8);
    }
}

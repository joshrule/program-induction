use rand::Rng;

use super::{SampleError, TRS};

impl TRS {
    /// Move a Rule from one place in the TRS to another at random, excluding the background.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use rand::{thread_rng};
    /// # use polytype::Context as TypeContext;
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule};
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some(".".to_string()));
    /// ops.push(ptp![0, 1; @arrow[tp!(@arrow[tp!(0), tp!(1)]), tp!(0), tp!(1)]]);
    /// sig.new_op(2, Some("PLUS".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// sig.new_op(1, Some("SUCC".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    /// sig.new_op(0, Some("ZERO".to_string()));
    /// ops.push(ptp![int]);
    ///
    /// let rules = vec![
    ///     parse_rule(&mut sig, "PLUS(x_ ZERO) = x_").expect("parsed rule"),
    ///     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))").expect("parsed rule"),
    /// ];
    ///
    /// let vars = vec![
    ///     ptp![int],
    ///     ptp![int],
    ///     ptp![int],
    /// ];
    ///
    /// println!("{:?}", sig.operators());
    /// for op in sig.operators() {
    ///     println!("{:?}/{}", op.name(&sig), op.arity())
    /// }
    /// for r in &rules {
    ///     println!("{:?}", r);
    /// }
    ///
    /// let ctx = TypeContext::default();
    /// let lexicon = Lexicon::from_signature(sig.clone(), ops, vars, vec![], vec![], false, ctx);
    ///
    /// let mut trs = TRS::new(&lexicon, rules).unwrap();
    ///
    /// let pretty_before = trs.to_string();
    ///
    /// let mut rng = thread_rng();
    ///
    /// let new_trs = trs.move_rule(&mut rng).expect("failed when moving rule");
    ///
    /// assert_ne!(pretty_before, new_trs.to_string());
    /// assert_eq!(new_trs.to_string(), "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_));\nPLUS(x_ ZERO) = x_;");
    /// # }
    /// ```
    pub fn move_rule<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let mut trs = self.clone();
        let num_rules = self.len();
        let background = &self.lex.0.read().expect("poisoned lexicon").background;
        let num_background = background.len();
        if num_background < num_rules - 1 {
            let i = rng.gen_range(num_background, num_rules);
            let mut j = rng.gen_range(num_background, num_rules);
            while j == i {
                j = rng.gen_range(num_background, num_rules);
            }
            trs.utrs.move_rule(i, j)?;
            Ok(trs)
        } else {
            Err(SampleError::OptionsExhausted)
        }
    }
}

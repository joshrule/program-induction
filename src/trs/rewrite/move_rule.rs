use rand::{seq::index::sample, Rng};

use super::{SampleError, TRS};

impl<'a, 'b> TRS<'a, 'b> {
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
    /// # use term_rewriting::{Signature, parse_rule};
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
    /// let lexicon = Lexicon::from_signature(sig.clone(), ops, vars, ctx);
    ///
    /// let mut trs = TRS::new(&lexicon, true, &[], rules).unwrap();
    ///
    /// let pretty_before = trs.to_string();
    ///
    /// let mut rng = thread_rng();
    ///
    /// let new_trs = trs.move_rule(&mut rng).expect("failed when moving rule");
    ///
    /// assert_ne!(pretty_before, new_trs.to_string());
    /// assert_eq!(new_trs.to_string(), "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_));\nPLUS(x_ ZERO) = x_;");
    /// ```
    pub fn move_rule<R: Rng>(&self, rng: &mut R) -> Result<TRS<'a, 'b>, SampleError> {
        let len = self.len();
        if len > 1 {
            let mut trs = self.clone();
            let idxs = sample(rng, len, 2);
            trs.utrs.move_rule(idxs.index(0), idxs.index(1))?;
            Ok(trs)
        } else {
            Err(SampleError::OptionsExhausted)
        }
    }
}

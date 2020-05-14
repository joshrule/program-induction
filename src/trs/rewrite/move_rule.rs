use rand::{seq::index::sample, Rng};

use super::{SampleError, TRS};

impl<'ctx, 'b> TRS<'ctx, 'b> {
    /// Move a Rule from one place in the TRS to another at random, excluding the background.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate term_rewriting;
    /// # extern crate rand;
    /// # use programinduction::trs::{parse_lexicon, parse_trs};
    /// # use rand::thread_rng;
    /// # use term_rewriting::Signature;
    /// # use polytype::{Source, atype::{with_ctx, TypeSchema, TypeContext}};
    /// with_ctx(10, |ctx: TypeContext<'_>| {
    ///     let mut rng = thread_rng();
    ///     let mut lex = parse_lexicon(
    ///         "PLUS/2: int -> int -> int; SUCC/1: int -> int; ZERO/0: int;",
    ///         &ctx,
    ///     ).expect("lex");
    ///
    ///     let trs1 = parse_trs(
    ///         "PLUS(v0_ ZERO) = v0_; PLUS(v0_ SUCC(v1_)) = SUCC(PLUS(v0_ v1_));",
    ///         &mut lex,
    ///         true,
    ///         &[]
    ///     ).expect("trs1");
    ///     let trs2 = trs1.move_rule(&mut rng).expect("trs2");
    ///
    ///     assert_eq!(trs1.to_string(), "PLUS(v0_ ZERO) = v0_;\nPLUS(v0_ SUCC(v1_)) = SUCC(PLUS(v0_ v1_));");
    ///     assert_eq!(trs2.to_string(), "PLUS(v0_ SUCC(v1_)) = SUCC(PLUS(v0_ v1_));\nPLUS(v0_ ZERO) = v0_;");
    /// })
    /// ```
    pub fn move_rule<R: Rng>(&self, rng: &mut R) -> Result<Self, SampleError> {
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

use std::collections::HashMap;
use term_rewriting::Rule;
use trs::{SampleError, TRS};

impl<'a, 'b> TRS<'a, 'b> {
    /// Compress the `TRS` by computing least general generalizations of its rules.
    pub fn lgg(&self) -> Result<TRS<'a, 'b>, SampleError> {
        if self.len() < 2 {
            return Err(SampleError::Subterm);
        }
        let mut l2 = self.utrs.clauses();
        l2.reverse();
        let mut l1 = Vec::with_capacity(l2.len());
        l1.push(l2.pop().unwrap());
        'next_rule: for r2 in l2.into_iter().rev() {
            for r1 in &mut l1 {
                if let Some(r3) = Rule::least_general_generalization(r1, &r2) {
                    *r1 = r3;
                    continue 'next_rule;
                }
            }
            l1.push(r2);
        }
        for r1 in l1.iter_mut() {
            r1.canonicalize(&mut HashMap::new());
        }
        let mut trs = self.clone();
        trs.utrs.rules = l1;
        Ok(trs)
    }
}

#[cfg(test)]
mod tests {
    use polytype::atype::{with_ctx, TypeContext};
    use trs::parser::{parse_lexicon, parse_trs};
    use trs::Lexicon;

    fn create_test_lexicon<'ctx, 'b>(ctx: &TypeContext<'ctx>) -> Lexicon<'ctx, 'b> {
        parse_lexicon(
            &[
                "C/0: list -> list;",
                "CONS/0: nat -> list -> list;",
                "NIL/0: list;",
                "DECC/0: nat -> int -> nat;",
                "DIGIT/0: int -> nat;",
                "./2: t1. t2. (t1 -> t2) -> t1 -> t2;",
                "0/0: int; 1/0: int; 2/0: int;",
                "3/0: int; 4/0: int; 5/0: int;",
                "6/0: int; 7/0: int; 8/0: int;",
                "9/0: int;",
            ]
            .join(" "),
            ctx,
        )
        .expect("parsed lexicon")
    }
    #[test]
    fn lgg_test() {
        with_ctx(1024, |ctx| {
            let mut lex = create_test_lexicon(&ctx);
            let trs = parse_trs(".(C .(.(CONS .(.(DECC .(DIGIT 5)) 4)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 0)) NIL)) = NIL; .(C .(.(CONS .(DIGIT 2)) var13_)) = .(.(CONS .(DIGIT 2)) .(C var13_)); .(C .(.(CONS .(DIGIT 0)) var20_)) = .(.(CONS .(DIGIT 0)) .(C var20_)); .(C .(.(CONS .(.(DECC .(DIGIT 7)) 7)) var19_)) = .(.(CONS .(.(DECC .(DIGIT 7)) 7)) .(C var19_)); .(C .(.(CONS .(.(DECC .(DIGIT 2)) 0)) var17_)) = .(.(CONS .(.(DECC .(DIGIT 2)) 0)) .(C var17_)); .(C .(.(CONS .(DIGIT 9)) var16_)) = .(.(CONS .(DIGIT 9)) .(C var16_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 0)) var15_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 0)) .(C var15_)); .(C .(.(CONS .(DIGIT 3)) var14_)) = .(.(CONS .(DIGIT 3)) .(C var14_)); .(C .(.(CONS .(.(DECC .(DIGIT 3)) 2)) var23_)) = .(.(CONS .(.(DECC .(DIGIT 3)) 2)) .(C var23_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 6)) var22_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 6)) .(C var22_));", &mut lex, true, &[]).expect("parsed trs");

            let maybe_trs = trs.lgg();
            assert!(maybe_trs.is_ok());

            let new_trs = maybe_trs.unwrap();
            println!("{}\n", new_trs);
            assert_eq!(new_trs.to_string(), ".(C .(.(CONS .(v0_ v1_)) NIL)) = NIL;\n.(C .(.(CONS .(v0_ v1_)) v2_)) = .(.(CONS .(v0_ v1_)) .(C v2_));");
        })
    }
}

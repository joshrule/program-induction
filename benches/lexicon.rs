extern crate criterion;
extern crate polytype;
extern crate programinduction;
extern crate rand;
extern crate term_rewriting;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use polytype::atype::{with_ctx, TypeContext, Variable as TVar};
use programinduction::trs::{
    parse_lexicon, parse_rulecontext, parse_term, Env, GenerationLimit, Lexicon, SampleParams,
};
use rand::{rngs::StdRng, SeedableRng};
use term_rewriting::{RuleContext, Term};

fn create_test_lexicon<'b, 'ctx>(ctx: &TypeContext<'ctx>) -> Lexicon<'ctx, 'b> {
    parse_lexicon(
        &[
            "C/0: list -> list;",
            "CONS/0: nat -> list -> list;",
            "EMPTY/0: list;",
            "HEAD/0: list -> nat;",
            "TAIL/0: list -> list;",
            "ISEMPTY/0: list -> bool;",
            "ISEQUAL/0: t1. t1 -> t1 -> bool;",
            "IF/0: t1. bool -> t1 -> t1 -> t1;",
            "TRUE/0: bool;",
            "FALSE/0: bool;",
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
    .expect("parsed test lexicon")
}

fn create_test_term(lex: &mut Lexicon) -> Term {
    parse_term(
        "(IF (ISEQUAL FALSE (ISEQUAL TRUE (ISEMPTY (TAIL (CONS (DIGIT 9) EMPTY))))) (CONS (HEAD (CONS (DIGIT 0) (CONS (DIGIT x_) EMPTY))) EMPTY) (C (CONS y_ (CONS z_ EMPTY))))",
        lex,
    )
        .expect("parsed test term")
}

fn create_test_rulecontext(lex: &mut Lexicon) -> RuleContext {
    parse_rulecontext(
        "(IF (ISEQUAL FALSE (ISEQUAL TRUE (ISEMPTY (TAIL (CONS [!] EMPTY))))) (CONS (HEAD (CONS (DIGIT 0) (CONS (DIGIT x_) [!]))) EMPTY) (C (CONS y_ (CONS z_ EMPTY)))) = [!] ([!] x_) (CONS z_ EMPTY) | (C (CONS y_ [!]))",
        lex,
    )
        .expect("parsed test term")
}

pub fn lexicon_enumerate_atoms_benchmark(c: &mut Criterion) {
    with_ctx(1024, |ctx| {
        let mut lex = create_test_lexicon(&ctx);
        let tp = ctx.intern_tvar(TVar(lex.lex.to_mut().src.fresh()));
        let mut env = Env::new(false, &lex, Some(lex.lex.src));
        c.bench_function("enumerate_atoms", |b| {
            b.iter(|| {
                let s = env.snapshot();
                black_box(&mut env).enumerate_atoms(black_box(tp)).count();
                env.rollback(s);
            })
        });
    })
}

pub fn lexicon_infer_term_benchmark(c: &mut Criterion) {
    with_ctx(1024, |ctx| {
        let mut lex = create_test_lexicon(&ctx);
        let term = create_test_term(&mut lex);
        c.bench_function("lexicon_infer_term", |b| {
            b.iter(|| black_box(&lex).infer_term(black_box(&term)))
        });
    })
}

pub fn lexicon_sample_term_benchmark(c: &mut Criterion) {
    with_ctx(1024, |ctx| {
        let mut lex = create_test_lexicon(&ctx);
        let schema = ctx.intern_monotype(ctx.intern_tvar(TVar(lex.lex.to_mut().src.fresh())));
        let params = SampleParams {
            atom_weights: (5.0, 5.0, 1.0, 1.0),
            variable: true,
            limit: GenerationLimit::TermSize(20),
        };
        let invent = true;
        let mut rng = StdRng::seed_from_u64(1);

        c.bench_function("lexicon_sample_term", |b| {
            b.iter(|| {
                black_box(&lex).sample_term(
                    black_box(&schema),
                    black_box(params),
                    black_box(invent),
                    black_box(&mut rng),
                )
            })
        });
    })
}

pub fn lexicon_logprior_term_benchmark(c: &mut Criterion) {
    with_ctx(1024, |ctx| {
        let mut lex = create_test_lexicon(&ctx);
        let term = create_test_term(&mut lex);
        let schema = ctx.intern_monotype(ctx.intern_tvar(TVar(lex.lex.to_mut().src.fresh())));
        let atom_weights = (5.0, 5.0, 1.0, 1.0);

        c.bench_function("lexicon_logprior_term", |b| {
            b.iter(|| {
                black_box(&lex).logprior_term(
                    black_box(&term),
                    black_box(&schema),
                    black_box(true),
                    black_box(atom_weights),
                )
            })
        });
    })
}

pub fn lexicon_infer_rulecontext_benchmark(c: &mut Criterion) {
    with_ctx(1024, |ctx| {
        let mut lex = create_test_lexicon(&ctx);
        let context = create_test_rulecontext(&mut lex);
        c.bench_function("lexicon_infer_rulecontext", |b| {
            b.iter(|| black_box(&lex).infer_rulecontext(black_box(&context)))
        });
    })
}

//pub fn lexicon_enumerate_terms_benchmark(c: &mut Criterion) {
//    let lex = create_test_lexicon();
//    let mut ctx = lex.context().clone();
//    let schema = TypeSchema::Monotype(ctx.new_variable());
//    let env = Environment::new(true);
//
//    c.bench_function("enumerate", |b| {
//        b.iter(|| {
//            black_box(&lex).enumerate_terms(
//                black_box(&schema),
//                black_box(5),
//                black_box(&env),
//                black_box(&ctx),
//            )
//        })
//    });
//}

criterion_group!(
    lexicon,
    lexicon_enumerate_atoms_benchmark,
    lexicon_infer_term_benchmark,
    lexicon_sample_term_benchmark,
    lexicon_logprior_term_benchmark,
    lexicon_infer_rulecontext_benchmark,
    //lexicon_enumerate_terms_benchmark,
);
criterion_main!(lexicon);

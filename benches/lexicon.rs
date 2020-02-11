extern crate criterion;
extern crate polytype;
extern crate programinduction;
extern crate rand;
extern crate term_rewriting;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use polytype::{Context as TypeContext, TypeSchema};
use programinduction::trs::{parse_lexicon, parse_term, Lexicon};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use term_rewriting::Term;

fn create_test_lexicon<'b>() -> Lexicon<'b> {
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
        TypeContext::default(),
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

pub fn lexicon_infer_term_benchmark(c: &mut Criterion) {
    let mut lex = create_test_lexicon();
    let term = create_test_term(&mut lex);
    let mut types = &mut HashMap::new();

    c.bench_function("lexicon_infer_term", |b| {
        b.iter(|| {
            Lexicon::infer_term(black_box(&lex), black_box(&term), black_box(&mut types)).drop()
        })
    });
}

pub fn lexicon_logprior_term_benchmark(c: &mut Criterion) {
    let mut lex = create_test_lexicon();
    let term = create_test_term(&mut lex);
    let schema = TypeSchema::Monotype(lex.fresh_type_variable());
    let atom_weights = (5.0, 5.0, 1.0, 1.0);
    let invent = true;

    c.bench_function("lexicon_logprior_term", |b| {
        b.iter(|| {
            Lexicon::logprior_term(
                black_box(&lex),
                black_box(&term),
                black_box(&schema),
                black_box(atom_weights),
                black_box(invent),
            )
        })
    });
}

pub fn lexicon_sample_term_benchmark(c: &mut Criterion) {
    let mut lex = create_test_lexicon();
    let schema = TypeSchema::Monotype(lex.fresh_type_variable());
    let atom_weights = (5.0, 5.0, 1.0, 1.0);
    let invent = true;
    let variable = true;
    let max_size = 20;
    let mut vars = vec![];
    let mut rng = StdRng::seed_from_u64(1);

    c.bench_function("lexicon_sample_term", |b| {
        b.iter(|| {
            Lexicon::sample_term(
                black_box(&mut lex),
                black_box(&schema),
                black_box(atom_weights),
                black_box(invent),
                black_box(variable),
                black_box(max_size),
                black_box(&mut vars),
                black_box(&mut rng),
            )
        })
    });
}

criterion_group!(
    lexicon,
    lexicon_infer_term_benchmark,
    lexicon_logprior_term_benchmark,
    lexicon_sample_term_benchmark,
);
criterion_main!(lexicon);

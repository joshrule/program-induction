#[macro_use]
extern crate criterion;
#[macro_use]
extern crate polytype;
extern crate programinduction;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use polytype::Context as TypeContext;
use programinduction::trs::{parse_lexicon, parse_rule, parse_trs, Lexicon, TRS};

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
    .expect("parsed lexicon")
}

fn create_variablize_test_lexicon<'b>() -> Lexicon<'b> {
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
        TypeContext::default(),
    )
    .expect("parsed lexicon")
}

pub fn trs_is_alpha_benchmark(c: &mut Criterion) {
    let mut lex = create_test_lexicon();
    let bg = vec![
        parse_rule("ISEMPTY EMPTY = TRUE", &mut lex).expect("parsed rule 1"),
        parse_rule("ISEMPTY (CONS x_ y_) = FALSE", &mut lex).expect("parsed rule 2"),
        parse_rule("ISEQUAL x_ x_ = TRUE", &mut lex).expect("parsed rule 3"),
        parse_rule("ISEQUAL x_ y_ = FALSE", &mut lex).expect("parsed rule 4"),
        parse_rule("HEAD (CONS x_ y_) = x_", &mut lex).expect("parsed rule 5"),
        parse_rule("IF TRUE  x_ y_ = x_", &mut lex).expect("parsed rule 6"),
        parse_rule("IF FALSE x_ y_ = y_", &mut lex).expect("parsed rule 7"),
        parse_rule("TAIL EMPTY = EMPTY", &mut lex).expect("parsed rule 8"),
        parse_rule("TAIL (CONS x_ y_) = y_", &mut lex).expect("parsed rule 9"),
    ];

    let mut lex1 = lex.clone();
    lex1.invent_operator(Some("F".to_string()), 0, &tp![@arrow[tp!(int), tp!(int)]]);
    let trs1 = parse_trs("F 0 = 1; F 1 = 0;", &mut lex1, true, &bg[..]).expect("parsed trs");

    let mut lex2 = lex.clone();
    lex2.invent_operator(Some("F".to_string()), 0, &tp![@arrow[tp!(int), tp!(int)]]);
    let trs2 = parse_trs("F 0 = 1; F 1 = 0;", &mut lex2, true, &bg[..]).expect("parsed trs");

    c.bench_function("is_alpha", |b| {
        b.iter(|| TRS::is_alpha(black_box(&trs1), black_box(&trs2)))
    });
}

pub fn trs_same_shape_benchmark(c: &mut Criterion) {
    let mut lex = create_test_lexicon();
    let bg = vec![
        parse_rule("ISEMPTY EMPTY = TRUE", &mut lex).expect("parsed rule 1"),
        parse_rule("ISEMPTY (CONS x_ y_) = FALSE", &mut lex).expect("parsed rule 2"),
        parse_rule("ISEQUAL x_ x_ = TRUE", &mut lex).expect("parsed rule 3"),
        parse_rule("ISEQUAL x_ y_ = FALSE", &mut lex).expect("parsed rule 4"),
        parse_rule("HEAD (CONS x_ y_) = x_", &mut lex).expect("parsed rule 5"),
        parse_rule("IF TRUE  x_ y_ = x_", &mut lex).expect("parsed rule 6"),
        parse_rule("IF FALSE x_ y_ = y_", &mut lex).expect("parsed rule 7"),
        parse_rule("TAIL EMPTY = EMPTY", &mut lex).expect("parsed rule 8"),
        parse_rule("TAIL (CONS x_ y_) = y_", &mut lex).expect("parsed rule 9"),
    ];

    let mut lex1 = lex.clone();
    lex1.invent_operator(Some("F".to_string()), 0, &tp![@arrow[tp!(int), tp!(int)]]);
    let trs1 = parse_trs("F 0 = 1; F 1 = 0;", &mut lex1, true, &bg[..]).expect("parsed trs");

    let mut lex2 = lex.clone();
    lex2.invent_operator(Some("F".to_string()), 0, &tp![@arrow[tp!(int), tp!(int)]]);
    let trs2 = parse_trs("F 0 = 1; F 1 = 0;", &mut lex2, true, &bg[..]).expect("parsed trs");

    c.bench_function("same_shape", |b| {
        b.iter(|| TRS::same_shape(black_box(&trs1), black_box(&trs2)))
    });
}

pub fn trs_variablize_benchmark(c: &mut Criterion) {
    let mut lex = create_variablize_test_lexicon();
    let trs = parse_trs(".(C .(.(CONS .(DIGIT 2)) var13_)) = .(.(CONS .(DIGIT 2)) .(C var13_)); .(C .(.(CONS .(DIGIT 0)) var20_)) = .(.(CONS .(DIGIT 0)) .(C var20_)); .(C .(.(CONS .(.(DECC .(DIGIT 7)) 7)) var19_)) = .(.(CONS .(.(DECC .(DIGIT 7)) 7)) .(C var19_)); .(C .(.(CONS .(.(DECC .(DIGIT 2)) 0)) var17_)) = .(.(CONS .(.(DECC .(DIGIT 2)) 0)) .(C var17_)); .(C .(.(CONS .(DIGIT 9)) var16_)) = .(.(CONS .(DIGIT 9)) .(C var16_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 0)) var15_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 0)) .(C var15_)); .(C .(.(CONS .(DIGIT 3)) var14_)) = .(.(CONS .(DIGIT 3)) .(C var14_)); .(C .(.(CONS .(.(DECC .(DIGIT 3)) 2)) var23_)) = .(.(CONS .(.(DECC .(DIGIT 3)) 2)) .(C var23_)); .(C .(.(CONS .(.(DECC .(DIGIT 1)) 6)) var22_)) = .(.(CONS .(.(DECC .(DIGIT 1)) 6)) .(C var22_));", &mut lex, true, &[]).expect("parsed trs");

    c.bench_function("variablize", |b| {
        b.iter(|| TRS::variablize(black_box(&trs)))
    });
}

criterion_group!(
    benches,
    trs_is_alpha_benchmark,
    trs_same_shape_benchmark,
    trs_variablize_benchmark,
);
criterion_main!(benches);

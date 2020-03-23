use super::lexicon::Lexicon;
use super::rewrite::TRS;
use nom;
use nom::types::CompleteStr;
use nom::{digit, Context as Nomtext, Err};
use polytype::{Context as TypeContext, TypeSchema};
use std::collections::HashMap;
use std::fmt;
use std::io;
use term_rewriting::{
    parse_context as parse_untyped_context, parse_rule as parse_untyped_rule,
    parse_rulecontext as parse_untyped_rulecontext, parse_term as parse_untyped_term, Atom,
    Context, Rule, RuleContext, Signature, Term,
};

#[derive(Debug, PartialEq)]
/// The error type for parsing operations.
pub struct ParseError;
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "failed parse")
    }
}
impl From<()> for ParseError {
    fn from(_: ()) -> ParseError {
        ParseError
    }
}
impl<'a> From<io::Error> for ParseError {
    fn from(_: io::Error) -> ParseError {
        ParseError
    }
}
impl<'a> From<Err<&'a str>> for ParseError {
    fn from(_: Err<&'a str>) -> ParseError {
        ParseError
    }
}
impl ::std::error::Error for ParseError {
    fn description(&self) -> &'static str {
        "parse error"
    }
}

/// Parse a [`Lexicon`], including its `background` knowledge, and any `templates`
/// that might be used during learning, ensuring that `background` and
/// `templates` typecheck according to the types in the newly instantiated
/// [`Lexicon`]. Syntax errors and failed typechecking both lead to `Err`.
///
/// # Lexicon syntax
///
/// `input` is parsed as a `lexicon`, defined below in [augmented Backus-Naur
/// form]. The definition of `schema` is as given in [`polytype`], while other
/// terms are as given in [`term_rewriting`]:
///
/// ```text
/// lexicon = *wsp *( *comment declaration ";" *comment ) *wsp
///
/// declaration = *wsp identifier *wsp ":" *wsp schema *wsp
/// ```
///
/// # Background syntax
///
/// `background` is parsed as a `rule_list`, defined below in [augmented Backus-Naur form].
/// The format of the other terms is as given in [`term_rewriting`]:
///
/// ```text
/// rule_list = *wsp *( *comment rule ";" *comment ) *wsp
/// ```
///
/// # Template Syntax
///
/// `templates` is parsed as a `rulecontext_list`, defined below in [augmented Backus-Naur form].
/// The format of the other terms is as given in [`term_rewriting`]:
///
/// ```text
/// rulecontext_list = *wsp *( *comment rulecontext ";" *comment ) *wsp
/// ```
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
/// [`polytype`]: ../../../polytype/index.html
/// [augmented Backus-Naur form]: https://en.wikipedia.org/wiki/Augmented_Backus–Naur_form
pub fn parse_lexicon<'a>(input: &str, ctx: TypeContext) -> Result<Lexicon<'a>, ParseError> {
    lexicon(CompleteStr(input), Signature::default(), vec![], ctx)
        .map(|(_, t)| t)
        .map_err(|_| ParseError)
}

/// Given a [`Lexicon`], parse and typecheck a [`TRS`]. The format of the
/// [`TRS`] is as given in [`term_rewriting`].
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`TRS`]: struct.TRS.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
pub fn parse_trs<'a, 'b>(
    input: &str,
    lex: &mut Lexicon<'b>,
    deterministic: bool,
    background: &'a [Rule],
) -> Result<TRS<'a, 'b>, ParseError> {
    match trs(CompleteStr(input), lex, deterministic, background) {
        Ok((CompleteStr(""), t)) => Ok(t),
        _ => Err(ParseError),
    }
}

/// Given a [`Lexicon`], parse and typecheck a [`Term`]. The format of the
/// [`Term`] is as given in [`term_rewriting`].
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`Term`]: ../../../term_rewriting/enum.Term.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
pub fn parse_term(input: &str, lex: &mut Lexicon) -> Result<Term, ParseError> {
    typed_term(CompleteStr(input), lex)
        .map(|(_, t)| t)
        .map_err(|_| ParseError)
}

/// Given a [`Lexicon`], parse and typecheck a [`Context`]. The format of the
/// [`Context`] is as given in [`term_rewriting`].
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`Context`]: ../../../term_rewriting/enum.Context.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
pub fn parse_context(input: &str, lex: &mut Lexicon) -> Result<Context, ParseError> {
    typed_context(CompleteStr(input), lex)
        .map(|(_, t)| t)
        .map_err(|_| ParseError)
}

/// Given a [`Lexicon`], parse and typecheck a [`RuleContext`]. The format of the
/// [`RuleContext`] is as given in [`term_rewriting`].
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`RuleContext`]: ../../../term_rewriting/struct.RuleContext.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
pub fn parse_rulecontext(input: &str, lex: &mut Lexicon) -> Result<RuleContext, ParseError> {
    typed_rulecontext(CompleteStr(input), lex)
        .map(|(_, t)| t)
        .map_err(|_| ParseError)
}

/// Given a [`Lexicon`], parse and typecheck a [`Rule`]. The format of the
/// [`Rule`] is as given in [`term_rewriting`].
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`Rule`]: ../../../term_rewriting/struct.Rule.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
pub fn parse_rule(input: &str, lex: &mut Lexicon) -> Result<Rule, ParseError> {
    typed_rule(input, lex)
        .map(|(_, t)| t)
        .map_err(|_| ParseError)
}

/// Given a [`Lexicon`], parse and typecheck a list of [`RuleContext`s]. The
/// format of a [`RuleContext`] is as given in [`term_rewriting`].
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`RuleContext`s]: ../../../term_rewriting/struct.RuleContext.html
/// [`RuleContext`]: ../../../term_rewriting/struct.RuleContext.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
pub fn parse_rulecontexts(input: &str, lex: &mut Lexicon) -> Result<Vec<RuleContext>, ParseError> {
    templates(CompleteStr(input), lex)
        .map(|(_, t)| t)
        .map_err(|_| ParseError)
}

/// Given a [`Lexicon`], parse and typecheck a list of [`Rule`s]. The
/// format of a [`Rule`] is as given in [`term_rewriting`].
///
/// [`Lexicon`]: ../struct.Lexicon.html
/// [`Rule`s]: ../../../term_rewriting/struct.Rule.html
/// [`Rule`]: ../../../term_rewriting/struct.Rule.html
/// [`term_rewriting`]: ../../../term_rewriting/index.html
pub fn parse_rules(input: &str, lex: &mut Lexicon) -> Result<Vec<Rule>, ParseError> {
    rules(CompleteStr(input), lex)
        .map(|(_, t)| t)
        .map_err(|_| ParseError)
}

fn schema_wrapper(input: CompleteStr) -> nom::IResult<CompleteStr, TypeSchema> {
    if let Ok(schema) = TypeSchema::parse(*input) {
        Ok((CompleteStr(""), schema))
    } else {
        Err(Err::Error(Nomtext::Code(input, nom::ErrorKind::Custom(0))))
    }
}

fn make_operator(
    (name, arity): (String, u8),
    sig: &mut Signature,
    schema: TypeSchema,
    ops: &mut Vec<TypeSchema>,
) -> Atom {
    let o = sig.new_op(arity, Some(name));
    ops.push(schema);
    Atom::Operator(o)
}

named!(colon<CompleteStr, CompleteStr>, tag!(":"));
named!(slash<CompleteStr, CompleteStr>, tag!("/"));
// reserved characters include:
// - [!] for holes
// -   (space) for separating terms
// - | for separating terms
// - # for comments
// - _ for variables
// - : for signatures
// - ( and ) for grouping
// - = for specifying rules
// - ; for ending statements
named!(identifier<CompleteStr, CompleteStr>, is_not!("[!]/| #_:()=;"));
named!(underscore<CompleteStr, CompleteStr>, tag!("_"));
named!(op_dec<CompleteStr, (String, u8)>,
        map!(do_parse!(ident: identifier >>
                       slash >>
                       arity: digit >>
                       (ident, arity)),
             |(s, a)| (s.to_string(), a.parse::<u8>().unwrap())));
named!(schema<CompleteStr, TypeSchema>,
       call!(schema_wrapper));
named!(comment<CompleteStr, CompleteStr>,
       map!(preceded!(tag!("#"), take_until_and_consume!("\n")),
            |s| CompleteStr(&s.trim())));
named_args!(declaration<'a>(sig: &mut Signature, ops: &mut Vec<TypeSchema>) <CompleteStr<'a>, (Atom, TypeSchema)>,
       map!(ws!(do_parse!(name: op_dec >>
                      colon >>
                      schema: schema >>
                      (name, schema))),
            |(n, s)| (make_operator(n, sig, s.clone(), ops), s)
       ));
fn lexicon<'a, 'b>(
    input: CompleteStr<'a>,
    mut sig: Signature,
    mut ops: Vec<TypeSchema>,
    ctx: TypeContext,
) -> nom::IResult<CompleteStr<'a>, Lexicon<'b>> {
    map!(
        input,
        ws!(many0!(do_parse!(
            many0!(ws!(comment))
                >> dec: take_until_and_consume!(";")
                >> expr_res!(declaration(dec, &mut sig, &mut ops))
                >> many0!(ws!(comment))
                >> ()
        ))),
        |_| Lexicon::from_signature(sig, ops, ctx)
    )
}
#[allow(clippy::cognitive_complexity)]
fn trs<'a, 'b, 'c>(
    input: CompleteStr<'a>,
    lex: &mut Lexicon<'c>,
    deterministic: bool,
    background: &'b [Rule],
) -> nom::IResult<CompleteStr<'a>, TRS<'b, 'c>> {
    ws!(
        input,
        do_parse!(
            many0!(ws!(comment))
                >> rules:
                    many0!(do_parse!(
                        many0!(ws!(comment))
                            >> rule_text: take_until_and_consume!(";")
                            >> rule: expr_res!(typed_rule(&rule_text, lex))
                            >> many0!(ws!(comment))
                            >> (rule.1)
                    ))
                >> (TRS::new_unchecked(lex, deterministic, background, rules))
        )
    )
}
fn typed_rule<'a>(input: &'a str, lex: &mut Lexicon) -> nom::IResult<CompleteStr<'a>, Rule> {
    lex.0.to_mut().signature.clear_variables();
    let result = parse_untyped_rule(&mut lex.0.to_mut().signature, input);
    if let Ok(rule) = result {
        let mut ctx = lex.0.ctx.clone();
        let result = lex.infer_rule(&rule, &mut HashMap::new(), &mut ctx);
        if result.is_ok() {
            return Ok((CompleteStr(""), rule));
        }
    }
    Err(Err::Error(Nomtext::Code(
        CompleteStr(input),
        nom::ErrorKind::Custom(0),
    )))
}
fn typed_term<'a>(
    input: CompleteStr<'a>,
    lex: &mut Lexicon,
) -> nom::IResult<CompleteStr<'a>, Term> {
    let result = parse_untyped_term(&mut lex.0.to_mut().signature, *input);
    if let Ok(term) = result {
        let mut ctx = lex.0.ctx.clone();
        if lex.infer_term(&term, &mut HashMap::new(), &mut ctx).is_ok() {
            return Ok((CompleteStr(""), term));
        }
    }
    Err(Err::Error(Nomtext::Code(input, nom::ErrorKind::Custom(0))))
}
fn typed_context<'a>(
    input: CompleteStr<'a>,
    lex: &mut Lexicon,
) -> nom::IResult<CompleteStr<'a>, Context> {
    let result = parse_untyped_context(&mut lex.0.to_mut().signature, *input);
    if let Ok(context) = result {
        let mut ctx = lex.0.ctx.clone();
        if lex
            .infer_context(&context, &mut HashMap::new(), &mut ctx)
            .is_ok()
        {
            return Ok((CompleteStr(""), context));
        }
    }
    Err(Err::Error(Nomtext::Code(input, nom::ErrorKind::Custom(0))))
}
fn typed_rulecontext<'a>(
    input: CompleteStr<'a>,
    lex: &mut Lexicon,
) -> nom::IResult<CompleteStr<'a>, RuleContext> {
    let result = parse_untyped_rulecontext(&mut lex.0.to_mut().signature, *input);
    if let Ok(rule) = result {
        let mut ctx = lex.0.ctx.clone();
        if lex
            .infer_rulecontext(&rule, &mut HashMap::new(), &mut ctx)
            .is_ok()
        {
            return Ok((CompleteStr(""), rule));
        }
    }
    Err(Err::Error(Nomtext::Code(input, nom::ErrorKind::Custom(0))))
}
named_args!(templates<'a>(lex: &mut Lexicon) <CompleteStr<'a>, Vec<RuleContext>>,
            ws!(do_parse!(templates: many0!(do_parse!(many0!(ws!(comment)) >>
                                                      rc_text: take_until_and_consume!(";") >>
                                                      rc: expr_res!(typed_rulecontext(rc_text, lex)) >>
                                                      many0!(ws!(comment)) >>
                                                      (rc.1))) >>
                          (templates)))
);

named_args!(rules<'a>(lex: &mut Lexicon) <CompleteStr<'a>, Vec<Rule>>,
            ws!(do_parse!(rules: many0!(do_parse!(many0!(ws!(comment)) >>
                                                      r_text: take_until_and_consume!(";") >>
                                                      r: expr_res!(typed_rule(&r_text, lex)) >>
                                                      many0!(ws!(comment)) >>
                                                      (r.1))) >>
                          (rules)))
);

#[cfg(test)]
mod tests {
    use super::*;
    use term_rewriting::Signature;

    #[test]
    fn comment_test() {
        let res = comment(CompleteStr("# this is a test\n"));
        assert_eq!(res.unwrap().1, CompleteStr("this is a test"));
    }

    #[test]
    fn declaration_op_test() {
        let mut ops = vec![];
        let mut sig = Signature::default();
        let (_, (a, s)) =
            declaration(CompleteStr("SUCC/1: int -> int"), &mut sig, &mut ops).unwrap();
        assert_eq!(a.display(&sig), "SUCC");
        assert_eq!(s.to_string(), "int → int");
    }

    #[test]
    fn lexicon_test() {
        let res = lexicon(
            CompleteStr("# COMMENT\nSUCC: int -> int;\nx_: list(int);"),
            Signature::default(),
            vec![],
            TypeContext::default(),
        );
        assert!(res.is_ok());
    }

    #[test]
    fn typed_rule_test() {
        let mut lex = lexicon(
            CompleteStr("ZERO/0: int; SUCC/1: int -> int;"),
            Signature::default(),
            vec![],
            TypeContext::default(),
        )
        .unwrap()
        .1;
        let res = typed_rule("SUCC(x_) = ZERO", &mut lex);
        let sig = &lex.0.signature;

        assert_eq!(res.unwrap().1.display(sig), "SUCC(v0_) = ZERO");
    }

    #[test]
    fn trs_test() {
        let mut lex = lexicon(
            CompleteStr("ZERO/0: int; SUCC/1: int -> int; PLUS/2: int -> int -> int;"),
            Signature::default(),
            vec![],
            TypeContext::default(),
        )
        .unwrap()
        .1;

        let res = trs(
            CompleteStr("PLUS(ZERO v0_) = ZERO; PLUS(SUCC(v0_) v1_) = SUCC(PLUS(v0_ v1_));"),
            &mut lex,
            true,
            &[],
        );
        let sig = &lex.0.signature;

        assert_eq!(
            res.unwrap().1.utrs.display(sig),
            "PLUS(ZERO v0_) = ZERO;\nPLUS(SUCC(v0_) v1_) = SUCC(PLUS(v0_ v1_));"
        );
    }

    #[test]
    fn context_test() {
        let mut lex = lexicon(
            CompleteStr("ZERO/0: int; SUCC/1: int -> int; PLUS/2: int -> int -> int;"),
            Signature::default(),
            vec![],
            TypeContext::default(),
        )
        .unwrap()
        .1;
        let res = typed_context(CompleteStr("PLUS(v0_ [!])"), &mut lex);
        let sig = &lex.0.signature;

        assert_eq!(res.unwrap().1.display(sig), "PLUS(v0_ [!])");
    }

    #[test]
    fn rulecontext_test() {
        let mut lex = lexicon(
            CompleteStr("ZERO/0: int; SUCC/1: int -> int; PLUS/2: int -> int -> int;"),
            Signature::default(),
            vec![],
            TypeContext::default(),
        )
        .unwrap()
        .1;
        let res = typed_rulecontext(CompleteStr("PLUS(v0_ [!]) = ZERO"), &mut lex);
        let sig = &lex.0.signature;

        assert_eq!(res.unwrap().1.display(sig), "PLUS(v0_ [!]) = ZERO");
    }

    #[test]
    fn templates_test() {
        let mut lex = lexicon(
            CompleteStr("ZERO/0: int; SUCC/1: int -> int; PLUS/2: int -> int -> int;"),
            Signature::default(),
            vec![],
            TypeContext::default(),
        )
        .unwrap()
        .1;
        let res = templates(
            CompleteStr("PLUS(v0_ [!]) = ZERO; [!] = SUCC(ZERO);"),
            &mut lex,
        );
        let sig = &lex.0.signature;

        let res_string = res
            .unwrap()
            .1
            .iter()
            .map(|rc| format!("{};", rc.display(sig)))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(res_string, "PLUS(v0_ [!]) = ZERO;\n[!] = SUCC(ZERO);");
    }
}

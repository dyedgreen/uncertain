#![feature(test)]
extern crate test;
use rand_distr::{Normal, Poisson, StandardNormal, Bernoulli};
use test::Bencher;
use uncertain::{Distribution, PointMass, Uncertain};

fn make_small() -> impl Uncertain<Value = f64> + Sized {
    let n = Distribution::from(StandardNormal);
    n.map(|n: f64| 5.0 * n)
}

#[bench]
fn small_eval_pr(b: &mut Bencher) {
    let u = make_small().map(|v| v > 0.5);
    b.iter(|| u.pr(0.5));
}

#[bench]
fn small_eval_expect(b: &mut Bencher) {
    let u = make_small();
    b.iter(|| u.expect(0.1));
}

#[bench]
fn big_eval_pr(bencher: &mut Bencher) {
    let a = Distribution::from(Poisson::new(10.7).unwrap()).into_ref();
    let b = Distribution::from(Normal::new(7.0, 3.0).unwrap());
    let n = Distribution::from(StandardNormal);

    let c = (&a).join(b, |a, b| a - 5.0 * b);
    let d = (&a)
        .join(n, |a, n: f64| a > 10.0 * n)
        .flat_map(|is_bigger| {
            Distribution::from(
                (if is_bigger {
                    Normal::new(-5.0, 1.0)
                } else {
                    Normal::new(5.0, 1.0)
                })
                .unwrap(),
            )
        });
    let u = c.join(d, |c, d| c + d).map(|v| v > -28.0);

    bencher.iter(|| u.pr(0.5));
}

#[bench]
fn big_eval_expect(bencher: &mut Bencher) {
    let a = Distribution::from(Poisson::new(10.7).unwrap()).into_ref();
    let b = Distribution::from(Normal::new(7.0, 3.0).unwrap());
    let n = Distribution::from(StandardNormal);

    let c = (&a).join(b, |a, b| a - 5.0 * b);
    let d = (&a)
        .join(n, |a, n: f64| a > 10.0 * n)
        .flat_map(|is_bigger| {
            Distribution::from(
                (if is_bigger {
                    Normal::new(-5.0, 1.0)
                } else {
                    Normal::new(5.0, 1.0)
                })
                .unwrap(),
            )
        });
    let u = c.join(d, |c, d| c + d);

    bencher.iter(|| u.expect(0.1));
}

fn make_boxed() -> impl Uncertain<Value = f64> + Sized {
    let choice = Distribution::from(Bernoulli::new(0.5).unwrap());
    choice.flat_map(|c| if c { PointMass::new(42.0).into_boxed() } else { Distribution::from(StandardNormal).into_boxed() })
}

#[bench]
fn boxed_eval_pr(b: &mut Bencher) {
    let u = make_boxed().map(|v| v > 10.0);
    b.iter(|| u.pr(0.5));
}

#[bench]
fn boxed_eval_expect(b: &mut Bencher) {
    let u = make_boxed();
    b.iter(|| u.expect(0.1));
}

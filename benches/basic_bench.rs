#![feature(test)]
extern crate test;
use rand_distr::{StandardNormal};
use test::{Bencher};
use uncertain::{Distribution, Uncertain};

#[bench]
fn simple_pr(b: &mut Bencher) {
    let gauss = Distribution::from(StandardNormal).map(|v: f64| v > 0.0);
    b.iter(|| gauss.pr(0.5));
}

#[bench]
fn simple_expect(b: &mut Bencher) {
    let gauss = Distribution::from(StandardNormal);
    b.iter(|| gauss.expect(0.01));
}

#[bench]
fn ref_pr(b: &mut Bencher) {
    let gauss = Distribution::from(StandardNormal).map(|v: f64| v > 0.0).into_ref();
    b.iter(|| gauss.pr(0.5));
}

#[bench]
fn ref_expect(b: &mut Bencher) {
    let gauss = Distribution::from(StandardNormal).map(|v: f64| v).into_ref();
    b.iter(|| gauss.expect(0.01));
}

#[bench]
fn boxed_pr(b: &mut Bencher) {
    let gauss = Distribution::from(StandardNormal).map(|v: f64| v > 0.0).into_boxed();
    b.iter(|| gauss.pr(0.5));
}

#[bench]
fn boxed_expect(b: &mut Bencher) {
    let gauss = Distribution::from(StandardNormal).into_boxed();
    b.iter(|| gauss.expect(0.01));
}

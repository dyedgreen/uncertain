use rand_distr::Bernoulli;
use uncertain::{Distribution, Uncertain};

#[test]
#[should_panic]
fn test_too_large_pr_panics() {
    let x = Distribution::from(Bernoulli::new(0.5).unwrap());
    x.pr(1.2);
}

#[test]
#[should_panic]
fn test_negative_pr_panics() {
    let x = Distribution::from(Bernoulli::new(0.5).unwrap());
    x.pr(-0.3);
}

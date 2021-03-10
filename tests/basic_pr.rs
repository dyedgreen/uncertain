use rand_distr::{Bernoulli, Binomial, Normal};
use uncertain::*;

#[test]
fn positive_pr() {
    let cases: Vec<f32> = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.89];
    for p in cases {
        let p_true = p + 0.1;
        let x = Distribution::from(Bernoulli::new(p_true.into()).unwrap());
        assert!(x.pr(p));
    }

    let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    for p in cases {
        let p_true_much_higher = p + 0.49;
        let x = Distribution::from(Bernoulli::new(p_true_much_higher.into()).unwrap());
        assert!(x.pr(p));
    }

    let cases: Vec<f32> = vec![0.1, 0.2, 0.3];
    for p in cases {
        let p_tru_way_higher = p + 0.6;
        let x = Distribution::from(Bernoulli::new(p_tru_way_higher.into()).unwrap());
        assert!(x.pr(p));
    }
}

#[test]
fn negative_pr() {
    let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
    for p in cases {
        let p_too_high = p + 0.1;
        let x = Distribution::from(Bernoulli::new(p.into()).unwrap());
        assert!(!x.pr(p_too_high));
    }

    let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
    for p in cases {
        let p_way_too_high = p + 0.2;
        let x = Distribution::from(Bernoulli::new(p.into()).unwrap());
        assert!(!x.pr(p_way_too_high));
    }

    let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    for p in cases {
        let p_very_way_too_high = p + 0.49;
        let x = Distribution::from(Bernoulli::new(p.into()).unwrap());
        assert!(!x.pr(p_very_way_too_high));
    }
}

#[test]
fn gaussian_pr() {
    let x = Distribution::from(Normal::new(5.0, 3.0).unwrap());
    let more_than_mean = x.map(|num| num > 5.0);

    assert!(more_than_mean.pr(0.1));
    assert!(more_than_mean.pr(0.2));
    assert!(more_than_mean.pr(0.3));
    assert!(more_than_mean.pr(0.4));

    assert!(!more_than_mean.pr(0.6));
    assert!(!more_than_mean.pr(0.7));
    assert!(!more_than_mean.pr(0.8));
    assert!(!more_than_mean.pr(0.9));
}

#[test]
fn very_certain() {
    let x = Distribution::from(Bernoulli::new(0.1).unwrap());
    assert!(x.pr(1e-5))
}

#[test]
fn not() {
    let x = Distribution::from(Bernoulli::new(0.7).unwrap());
    assert!(x.pr(0.2));
    assert!(x.pr(0.6));
    let not_x = x.not();
    assert!(not_x.pr(0.2));
    assert!(!not_x.pr(0.6));
}

#[test]
fn sampling_sanity_check() {
    let x = Distribution::from(Binomial::new(100, 0.5).unwrap()).into_ref();
    let diff = (&x).sub(&x);
    assert!(diff.map(|d| d == 0).pr(0.9999));
}

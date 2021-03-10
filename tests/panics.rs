use uncertain::{PointMass, Uncertain};

#[test]
#[should_panic]
fn test_too_large_pr_panics() {
    let x = PointMass::new(true);
    x.pr(1.2);
}

#[test]
#[should_panic]
fn test_negative_pr_panics() {
    let x = PointMass::new(true);
    x.pr(-0.3);
}

#[test]
#[should_panic]
fn test_nagative_expect_precision_panics() {
    let x = PointMass::new(0.0);
    x.expect(-0.1).ok();
}

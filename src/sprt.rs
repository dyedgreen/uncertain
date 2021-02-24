use crate::Uncertain;
use rand::Rng;

const D0: f32 = 0.999;
const D1: f32 = 0.999;

const STEP: usize = 10;
const MAXS: usize = 1000;

fn accept_likelyhood(prob: f32, val: bool) -> f32 {
    let p = 0.5 * (1.0 + prob);
    if val {
        p
    } else {
        1.0 - p
    }
}

fn reject_likelyhood(prob: f32, val: bool) -> f32 {
    let p = 0.5 * prob;
    if val {
        p
    } else {
        1.0 - p
    }
}

fn log_likelyhood_ratio(prob: f32, val: bool) -> f32 {
    reject_likelyhood(prob, val).ln() - accept_likelyhood(prob, val).ln()
}

pub fn sequential_probability_ratio_test<U, R>(prob: f32, src: &U, rng: &mut R) -> bool
where
    U: Uncertain + ?Sized,
    U::Value: Into<bool>,
    R: Rng,
{
    let upper_ln = (D1 / (1.0 - D1)).ln();
    let lower_ln = ((1.0 - D0) / D0).ln();

    let val = src.sample(rng, 0).into();
    let mut ratio_ln = log_likelyhood_ratio(prob, val);

    for step in 0..MAXS {
        for s in 0..STEP {
            let epoch = STEP * step + s;
            let val = src.sample(rng, epoch).into();
            ratio_ln += log_likelyhood_ratio(prob, val);
        }
        if ratio_ln > upper_ln || ratio_ln < lower_ln {
            break;
        }
    }

    ratio_ln < lower_ln
}

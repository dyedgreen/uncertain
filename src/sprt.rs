use crate::Uncertain;
use rand_pcg::Pcg32;

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

/// Compute the sequential probability ration test.
pub fn compute<U>(src: &U, prob: f32) -> bool
where
    U: Uncertain + ?Sized,
    U::Value: Into<bool>,
{
    let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);

    let upper_ln = (D1 / (1.0 - D1)).ln();
    let lower_ln = ((1.0 - D0) / D0).ln();
    let mut ratio_ln = 0.0;

    for batch in 0..MAXS {
        for batch_step in 0..STEP {
            let epoch = STEP * batch + batch_step;
            let val = src.sample(&mut rng, epoch).into();
            ratio_ln += log_likelyhood_ratio(prob, val);
        }
        if ratio_ln > upper_ln || ratio_ln < lower_ln {
            break;
        }
    }

    ratio_ln < lower_ln
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use rand_distr::Bernoulli;

    #[test]
    fn basic_sprt_works() {
        let src = Distribution::from(Bernoulli::new(0.5).unwrap());

        assert!(compute(&src, 0.4));
        assert!(!compute(&src, 0.6));
    }

    #[test]
    fn likelyhood_sanity_check() {
        assert_eq!(accept_likelyhood(0.0, true), 0.5);
        assert_eq!(accept_likelyhood(1.0, true), 1.0);
        assert_eq!(reject_likelyhood(0.0, false), 1.0);
        assert_eq!(reject_likelyhood(1.0, false), 0.5);
    }
}

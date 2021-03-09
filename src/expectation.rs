use crate::Uncertain;
use rand_pcg::Pcg32;
use std::error::Error;
use std::fmt;

const STEP: usize = 10;
const MAXS: usize = 1000;

#[derive(Debug, Clone)]
pub struct ConvergenceError {
    sample_mean: f64,
    diff_sum: f64,
    steps: f64,
    precision: f64,
}

impl ConvergenceError {
    // TODO
}

impl fmt::Display for ConvergenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let std = mean_standard_deviation(self.diff_sum, self.steps);
        // TODO
        write!(
            f,
            "Expected value {} +/- {} did not converge to desired precision {}",
            self.sample_mean, std, self.precision
        )
    }
}

impl Error for ConvergenceError {}

fn sample_variance(diff_sum: f64, steps: f64) -> f64 {
    diff_sum / steps // = sigma^2 i.e. var(x)
}

fn mean_standard_deviation(diff_sum: f64, steps: f64) -> f64 {
    diff_sum.sqrt() / steps // = sqrt( sigma^2 / n ) i.e. sqrt(var(E(x)))
}

/// Compute the sample expectation.
pub fn compute<U>(src: &U, precision: f64) -> Result<f64, ConvergenceError>
where
    U: Uncertain + ?Sized,
    U::Value: Into<f64>,
{
    let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);

    let mut sample_mean = 0.0;
    let mut diff_sum = 0.0;
    let mut steps = 0.0;

    for _ in 0..MAXS {
        for _ in 0..STEP {
            let sample = src.sample(&mut rng, steps as usize).into();
            let prev_sample_mean = sample_mean;

            // Using Welford's online algorithm:
            // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            steps += 1.0;
            sample_mean = prev_sample_mean + (sample - prev_sample_mean) / steps;
            diff_sum = diff_sum + (sample - prev_sample_mean) * (sample - sample_mean);
        }

        if mean_standard_deviation(diff_sum, steps) <= precision {
            return Ok(sample_mean);
        }
    }

    Err(ConvergenceError {
        sample_mean,
        diff_sum,
        steps,
        precision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Distribution;
    use rand_distr::Normal;

    #[test]
    fn simple_expectation() {
        let values = vec![0.0, 1.0, 5.0, 17.0, 23525.108213];
        for val in values {
            let x = Distribution::from(Normal::new(val, 1.0).unwrap());

            let mu = compute(&x, 0.1);
            assert!(mu.is_ok());
            assert!((mu.unwrap() - val).abs() < 0.1);
        }
    }

    #[test]
    fn failed_expectation() {
        let x = Distribution::from(Normal::new(0.0, 1000.0).unwrap());

        let mu = compute(&x, 0.1);
        assert!(mu.is_err());

        let mu = compute(&x, 100.0);
        assert!(mu.is_ok());
        assert!(mu.unwrap().abs() < 100.0);
    }
}

use rand_distr::{Normal, Poisson};
use uncertain::*;

#[test]
fn basic_gaussian() {
    let centers: Vec<f64> = vec![0.0, 3.5, 1.73, 234.23235];
    for center in centers {
        let x = Distribution::from(Normal::new(center, 1.0).unwrap());
        let mu = x.expect(0.1);
        assert!(mu.is_ok());
        assert!((mu.unwrap() - center).abs() < 0.1);
    }
    let centers: Vec<f32> = vec![0.0, 3.5, 1.73, 234.23235];
    for center in centers {
        let x = Distribution::from(Normal::new(center, 1.0).unwrap());
        let mu = x.expect(0.1);
        assert!(mu.is_ok());
        assert!((mu.unwrap() - center).abs() < 0.1);
    }
}

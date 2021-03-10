use rand_distr::{Normal, Poisson};
use rand_pcg::Pcg32;
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

#[test]
fn basic_poisson() {
    let centers: Vec<f64> = vec![1.0, 2.0, 3.0, 5.0, 15.0];
    for center in centers {
        let x = Distribution::from(Poisson::new(center).unwrap());
        let mu = x.expect(0.1);
        assert!(mu.is_ok());
        assert!((mu.unwrap() - center).abs() < 0.1);
    }

    let centers: Vec<f64> = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    for center in centers {
        let x = Distribution::from(Poisson::new(center).unwrap());
        let mu = x.expect(10.0);
        assert!(mu.is_ok());
        assert!((mu.unwrap() - center).abs() < 10.0);
    }

    let centers: Vec<f64> = vec![0.35235, 1.234, 4.324234, 6.3435];
    for center in centers {
        let x = Distribution::from(Poisson::new(center).unwrap());
        let mu = x.expect(0.1);
        assert!(mu.is_ok());
        assert!((mu.unwrap() - center).abs() < 0.1);
    }
}

#[test]
fn very_easy() {
    let x = PointMass::new(5.0);
    assert_eq!(x.expect(0.01).unwrap(), 5.0);

    let x = Distribution::from(Normal::new(5.0, 2.0).unwrap()).into_ref();
    let y = (&x).sub(&x);
    assert_eq!(y.expect(0.01).unwrap(), 0.0);
}

#[test]
fn very_hard() {
    struct UncertainCounter;
    impl Uncertain for UncertainCounter {
        type Value = f64;

        fn sample(&self, _: &mut Pcg32, epoch: usize) -> Self::Value {
            epoch as f64
        }
    }

    let c = UncertainCounter;
    let mu = c.expect(0.1); // should not converge (!)
    assert!(mu.is_err());
}

#[test]
fn composite_gaussian_poisson() {
    let g = Distribution::from(Normal::new(5.0, 2.0).unwrap());
    let p = Distribution::from(Poisson::new(3.0).unwrap());
    let s = g.add(p);

    let mu = s.expect(0.1 as f64);
    assert!(mu.is_ok());
    assert!((mu.unwrap() - 8.0).abs() < 0.1);
}

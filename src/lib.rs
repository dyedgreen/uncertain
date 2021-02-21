use rand::{distributions::Distribution, Rng, SeedableRng};
use rand_distr::Normal;
use rand_pcg::{Pcg32, Pcg64};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

pub trait Uncertain<T, R = Pcg32>
where
    R: Rng + ?Sized,
{
    fn sample(&self, rng: &mut R, epoch: usize) -> T;

    fn into_boxed(self) -> BoxedUncertain<T, R>
    where
        Self: 'static + Sized,
        T: Clone,
    {
        BoxedUncertain {
            ptr: Rc::new(self),
            cache: Rc::new(RefCell::new(None)),
        }
    }
}

pub trait UncertainOps<T, R>: Uncertain<T, R>
where
    R: Rng + ?Sized,
{
    fn comp<O, U, C, F>(self, other: U, func: F) -> UncertainComputation<T, O, C, F, R, Self, U>
    where
        Self: Sized,
        U: Uncertain<O, R>,
        F: Fn(T, O) -> C,
    {
        UncertainComputation {
            a: self,
            b: other,
            func,
            _p: PhantomData {},
        }
    }

    fn add<O, U>(self, other: U) -> UncertainSum<T, O, R, Self, U>
    where
        Self: Sized,
        T: std::ops::Add<O>,
        U: Uncertain<O, R>,
    {
        UncertainSum {
            a: self,
            b: other,
            _p: PhantomData {},
        }
    }
}

impl<T, R, U> UncertainOps<T, R> for U
where
    R: Rng + ?Sized,
    U: Uncertain<T, R>,
{
}

struct Reusable<T, R: Rng + ?Sized> {
    src: Box<dyn Uncertain<T, R>>,
    cache: RefCell<Option<(usize, T)>>,
}

#[derive(Clone)]
pub struct BoxedUncertain<T, R>
where
    R: Rng + ?Sized,
    T: Clone,
{
    ptr: Rc<dyn Uncertain<T, R>>,
    cache: Rc<RefCell<Option<(usize, T)>>>,
}

impl<T: Clone, R: Rng + ?Sized> Uncertain<T, R> for BoxedUncertain<T, R> {
    fn sample(&self, rng: &mut R, epoch: usize) -> T {
        let mut cache = self.cache.borrow_mut();
        if let Some((last_epoch, last_value)) = &*cache {
            if *last_epoch == epoch {
                return last_value.clone();
            }
        }
        let value = self.ptr.sample(rng, epoch);
        *cache = Some((epoch, value.clone()));
        value
    }
}

pub struct UncertainDistribution<T, D>
where
    D: Distribution<T>,
{
    dist: D,
    _p: PhantomData<T>,
}

impl<T, R, D> Uncertain<T, R> for UncertainDistribution<T, D>
where
    R: Rng + ?Sized,
    D: Distribution<T>,
{
    fn sample(&self, rng: &mut R, _epoch: usize) -> T {
        self.dist.sample(rng)
    }
}

impl<T, D> From<D> for UncertainDistribution<T, D>
where
    D: Distribution<T>,
{
    fn from(dist: D) -> Self {
        Self {
            dist,
            _p: PhantomData {},
        }
    }
}

pub struct UncertainComputation<A, B, C, F, R, UA, UB>
where
    F: Fn(A, B) -> C,
    R: Rng + ?Sized,
    UA: Uncertain<A, R>,
    UB: Uncertain<B, R>,
{
    a: UA,
    b: UB,
    func: F,
    _p: PhantomData<(A, B, C, R)>,
}

impl<A, B, C, F, R, UA, UB> Uncertain<C, R> for UncertainComputation<A, B, C, F, R, UA, UB>
where
    F: Fn(A, B) -> C,
    R: Rng + ?Sized,
    UA: Uncertain<A, R>,
    UB: Uncertain<B, R>,
{
    fn sample(&self, rng: &mut R, epoch: usize) -> C {
        let a = self.a.sample(rng, epoch);
        let b = self.b.sample(rng, epoch);
        (self.func)(a, b)
    }
}

pub struct UncertainSum<A, B, R, UA, UB>
where
    A: std::ops::Add<B>,
    R: Rng + ?Sized,
    UA: Uncertain<A, R>,
    UB: Uncertain<B, R>,
{
    a: UA,
    b: UB,
    _p: PhantomData<(A, B, R)>,
}

impl<A, B, R, UA, UB> Uncertain<A::Output, R> for UncertainSum<A, B, R, UA, UB>
where
    A: std::ops::Add<B>,
    R: Rng + ?Sized,
    UA: Uncertain<A, R>,
    UB: Uncertain<B, R>,
{
    fn sample(&self, rng: &mut R, epoch: usize) -> A::Output {
        let a = self.a.sample(rng, epoch);
        let b = self.b.sample(rng, epoch);
        a + b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let a: UncertainDistribution<f32, _> = Normal::new(0.0, 1.0).unwrap().into();
        let b: UncertainDistribution<f32, _> = Normal::new(5.0, 1.0).unwrap().into();
        let c = a.add(b);

        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        for epoch in 0..10 {
            println!("{:?}", c.sample(&mut rng, epoch));
        }
        assert!(false);
    }

    #[test]
    fn clone_shares_values() {
        let x: UncertainDistribution<f32, _> = Normal::new(10.0, 1.0).unwrap().into();
        let x = x.into_boxed();
        let y = x.clone();
        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        for epoch in 0..1000 {
            assert_eq!(x.sample(&mut rng, epoch), y.sample(&mut rng, epoch));
        }
    }

    #[test]
    fn reused_add() {
        let x: UncertainDistribution<f32, _> = Normal::new(10.0, 1.0).unwrap().into();
        let y: UncertainDistribution<f32, _> = Normal::new(5.0, 1.0).unwrap().into();
        let x = x.into_boxed();
        let a = x.clone().add(y);
        let b = a.add(x);

        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        for epoch in 0..10 {
            println!("{:?}", b.sample(&mut rng, epoch));
        }
        assert!(false);
    }
}

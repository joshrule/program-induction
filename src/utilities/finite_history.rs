use std::collections::VecDeque;

/// Track a finite number of the most recently seen items in a stream.
pub struct FiniteHistory<T> {
    // The number of items.
    size: usize,
    // The items.
    data: VecDeque<T>,
}

impl<T> FiniteHistory<T> {
    /// Create a new `FiniteHistory` that can hold `size` items.
    pub fn new(size: usize) -> Self {
        let mut data = VecDeque::new();
        data.reserve_exact(size);
        FiniteHistory { size, data }
    }
    /// Add an `item`, removing another if necessary.
    pub fn add(&mut self, item: T) {
        if self.data.len() == self.size {
            self.data.pop_front();
        }
        self.data.push_back(item);
    }
}
impl<T> FiniteHistory<T>
where
    for<'a> &'a T: Into<f64>,
{
    /// Compute the mean value in the history.
    pub fn mean(&self) -> f64 {
        self.data.iter().map(|x| x.into()).sum::<f64>() / (self.size as f64)
    }
}

/// A wrapper for `bool` that can be converted to `f64`.
pub struct FHBool(pub bool);

impl<'a> From<&'a FHBool> for f64 {
    fn from(b: &FHBool) -> Self {
        if b.0 {
            1.0
        } else {
            0.0
        }
    }
}

use std::collections::VecDeque;

/// Track a finite number of the most recently seen items in a stream.
pub struct FiniteHistory<T> {
    // The number of items.
    size: usize,
    // The items.
    data: VecDeque<T>,
    // The total number of items seen.
    n: usize,
}

impl<T> FiniteHistory<T> {
    /// Create a new `FiniteHistory` that can hold `size` items.
    pub fn new(size: usize) -> Self {
        let mut data = VecDeque::new();
        data.reserve_exact(size);
        FiniteHistory { size, data, n: 0 }
    }
    /// Add an `item`, removing another if necessary.
    pub fn add(&mut self, item: T) {
        self.n += 1;
        if self.data.len() == self.size {
            self.data.pop_front();
        }
        self.data.push_back(item);
    }
    pub fn n(&self) -> usize {
        self.n
    }
}

// adapted from: https://stackoverflow.com/questions/34247038
impl<T> FiniteHistory<T>
where
    T: Into<f64> + Copy,
{
    /// Compute the mean value in the history.
    pub fn mean(&self) -> f64 {
        let mut count = 0.0;
        self.data
            .iter()
            .copied()
            .map(Into::into)
            .inspect(|_| count += 1.0)
            .sum::<f64>()
            / count
    }
}

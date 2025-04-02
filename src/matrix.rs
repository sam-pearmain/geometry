use std::ops::{Index, IndexMut};
use std::fmt::{Debug, Display};
use num::{Zero, One};

#[allow(unused)]

pub trait Tensor: Index<Self::Index> + IndexMut<Self::Index> + Debug + Display {
    type Index;
}

#[derive(PartialEq, PartialOrd)]
pub struct StaticMatrix<T, const ROWS: usize, const COLS: usize> {
    data: [T],
}

pub struct Matrix<T> {
    pub rows: usize,
    pub cols: usize, 
    data: Vec<T>,
}

impl<T: Default + Clone> Matrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![T::default(); rows * cols] }
    }
}

impl<T: Zero + Clone> Matrix<T> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![T::zero(); rows * cols] }
    }
}

impl<T: One + Clone> Matrix<T> {
    pub fn ones(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![T::one(); rows * cols] }
    }
}

impl<T: Zero + One + Clone> Matrix<T> {
    pub fn identity(dims: usize) -> Self {
        let mut matrix = Self::zeros(dims, dims);
        for i in 0..dims {
            matrix[(i, i)] = T::one();
        }
        matrix
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.cols + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.cols + index.1]
    }
} 

impl<T> Debug for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "matrix::{} [{}x{}]", std::any::type_name::<T>(), self.rows, self.cols)?;
        Ok(())
    }
}

impl<T: Debug> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, " {:?} ", self[(i, j)])?;
            }
            write!(f, "\n")?;
        }
        Ok(())

    }
}

impl<T: Debug> Tensor for Matrix<T> {
    type Index = (usize, usize);
}
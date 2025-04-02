#![allow(unused)]

use std::ops::{Index, IndexMut};
use std::fmt::{Debug, Display};
use num::{Zero, One};

pub trait Tensor: Index<Self::Index> + IndexMut<Self::Index> + Debug + Display {
    type Index;
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

pub struct StaticMatrix<T, const ROWS: usize, const COLS: usize> {
    data: [[T; COLS]; ROWS]
}

impl<T: Default + Copy, const ROWS: usize, const COLS: usize> StaticMatrix<T, ROWS, COLS> {
    pub fn new() -> Self {
        Self {
            data: [[T::default(); COLS]; ROWS]
        }
    }
}

impl<T: Zero + Copy, const ROWS: usize, const COLS: usize> StaticMatrix<T, ROWS, COLS> {
    pub fn zeros() -> Self {
        Self {
            data: [[T::zero(); COLS]; ROWS]
        }
    }
}

impl<T: One + Copy, const ROWS: usize, const COLS: usize> StaticMatrix<T, ROWS, COLS> {
    pub fn ones() -> Self {
        Self {
            data: [[T::one(); COLS]; ROWS]
        }
    }
}

impl<T: Zero + One + Copy, const N: usize> StaticMatrix<T, N, N> {
    pub fn identity() -> Self {
        let mut matrix = Self::zeros();
        for i in 0..N {
            matrix[(i, i)] = T::one();
        }
        matrix
    }
}

impl<T, const ROWS: usize, const COLS: usize> Index<(usize, usize)> for StaticMatrix<T, ROWS, COLS> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}

impl<T: Copy, const ROWS: usize, const COLS: usize> IndexMut<(usize, usize)> for StaticMatrix<T, ROWS, COLS> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0][index.1]
    }
}

impl<T, const ROWS: usize, const COLS: usize> Debug for StaticMatrix<T, ROWS, COLS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "static_matrix::{} [{}x{}]", std::any::type_name::<T>(), ROWS, COLS)
    }
}

impl<T: Debug, const ROWS: usize, const COLS: usize> Display for StaticMatrix<T, ROWS, COLS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..ROWS {
            for j in 0..COLS {
                write!(f, " {:?} ", self[(i, j)])?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl<T: Debug + Copy, const ROWS: usize, const COLS: usize> Tensor for StaticMatrix<T, ROWS, COLS> {
    type Index = (usize, usize);
}
use num_traits::NumCast;
use pyo3::{prelude::*, IntoPyObject, IntoPyObjectExt};
use ndarray::{Array1, Array2, s};
use numpy::{PyArray2, PyArray3, Element};
use crate::metadata::{Data, Dtype};

pub trait NumpyElement: Element + NumCast {}
impl<T: Element + NumCast> NumpyElement for T {}

pub trait IntoPyObjectShaped<'py> {
    type Target;
    type Output;
    type Error;

    fn into_pyobject_shaped(self, py: Python<'py>, width: usize, height: usize) 
        -> Result<Self::Output, Self::Error>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum FieldData {
    U8(Array2<u8>),
    U16(Array2<u16>),
    U32(Array2<u32>),
    U64(Array2<u64>),
    I8(Array2<i8>),
    I16(Array2<i16>),
    I32(Array2<i32>),
    I64(Array2<i64>),
    F32(Array2<f32>),
    F64(Array2<f64>),
}

impl FieldData {
    pub fn new(dtype: Dtype, npoints: usize, count: usize) -> Self {
        match dtype {
            Dtype::U8 => FieldData::U8(Array2::zeros((npoints, count))),
            Dtype::U16 => FieldData::U16(Array2::zeros((npoints, count))),
            Dtype::U32 => FieldData::U32(Array2::zeros((npoints, count))),
            Dtype::U64 => FieldData::U64(Array2::zeros((npoints, count))),
            Dtype::I8 => FieldData::I8(Array2::zeros((npoints, count))),
            Dtype::I16 => FieldData::I16(Array2::zeros((npoints, count))),
            Dtype::I32 => FieldData::I32(Array2::zeros((npoints, count))),
            Dtype::I64 => FieldData::I64(Array2::zeros((npoints, count))),
            Dtype::F32 => FieldData::F32(Array2::zeros((npoints, count))),
            Dtype::F64 => FieldData::F64(Array2::zeros((npoints, count))),
        }
    }

    /// Return the length (total number of values) in this field.
    pub fn len(&self) -> usize {
        match self {
            FieldData::U8(arr) => arr.len(),
            FieldData::U16(arr) => arr.len(),
            FieldData::U32(arr) => arr.len(),
            FieldData::U64(arr) => arr.len(),
            FieldData::I8(arr) => arr.len(),
            FieldData::I16(arr) => arr.len(),
            FieldData::I32(arr) => arr.len(),
            FieldData::I64(arr) => arr.len(),
            FieldData::F32(arr) => arr.len(),
            FieldData::F64(arr) => arr.len(),
        }
    }

    /// Return the number of points in this field.
    pub fn npoints(&self) -> usize {
        match self {
            FieldData::U8(arr) => arr.shape()[0],
            FieldData::U16(arr) => arr.shape()[0],
            FieldData::U32(arr) => arr.shape()[0],
            FieldData::U64(arr) => arr.shape()[0],
            FieldData::I8(arr) => arr.shape()[0],
            FieldData::I16(arr) => arr.shape()[0],
            FieldData::I32(arr) => arr.shape()[0],
            FieldData::I64(arr) => arr.shape()[0],
            FieldData::F32(arr) => arr.shape()[0],
            FieldData::F64(arr) => arr.shape()[0],
        }
    }

    /// Return the number of columns in this field.
    pub fn count(&self) -> usize {
        match self {
            FieldData::U8(arr) => arr.shape()[1],
            FieldData::U16(arr) => arr.shape()[1],
            FieldData::U32(arr) => arr.shape()[1],
            FieldData::U64(arr) => arr.shape()[1],
            FieldData::I8(arr) => arr.shape()[1],
            FieldData::I16(arr) => arr.shape()[1],
            FieldData::I32(arr) => arr.shape()[1],
            FieldData::I64(arr) => arr.shape()[1],
            FieldData::F32(arr) => arr.shape()[1],
            FieldData::F64(arr) => arr.shape()[1],
        }
    }

    /// Return the data type of this field.
    pub fn dtype(&self) -> Dtype {
        match self {
            FieldData::U8(_) => Dtype::U8,
            FieldData::U16(_) => Dtype::U16,
            FieldData::U32(_) => Dtype::U32,
            FieldData::U64(_) => Dtype::U64,
            FieldData::I8(_) => Dtype::I8,
            FieldData::I16(_) => Dtype::I16,
            FieldData::I32(_) => Dtype::I32,
            FieldData::I64(_) => Dtype::I64,
            FieldData::F32(_) => Dtype::F32,
            FieldData::F64(_) => Dtype::F64,
        }
    }

    // Return the data as a 2D array of the specified type.
    pub fn get_data<A: Data + NumCast>(&self) -> Array2<A> {
        match self {
            FieldData::U8(arr) => arr.mapv(|x| A::from(x).unwrap()),
            FieldData::U16(arr) => arr.mapv(|x| A::from(x).unwrap()),
            FieldData::U32(arr) => arr.mapv(|x| A::from(x).unwrap()),
            FieldData::U64(arr) => arr.mapv(|x| A::from(x).unwrap()),
            FieldData::I8(arr) => arr.mapv(|x| A::from(x).unwrap()),
            FieldData::I16(arr) => arr.mapv(|x| A::from(x).unwrap()),
            FieldData::I32(arr) => arr.mapv(|x| A::from(x).unwrap()),
            FieldData::I64(arr) => arr.mapv(|x| A::from(x).unwrap()),
            FieldData::F32(arr) => arr.mapv(|x| A::from(x).unwrap()),
            FieldData::F64(arr) => arr.mapv(|x| A::from(x).unwrap()),
        }
    }

    /// Return a single row of data as a 1D array of the specified type.
    pub fn get_row<A: Data + NumCast>(&self, row_idx: usize) -> Array1<A> {
        match self {
            FieldData::U8(arr) => arr.slice(s![row_idx, ..]).mapv(|x| A::from(x).unwrap()),
            FieldData::U16(arr) => arr.slice(s![row_idx, ..]).mapv(|x| A::from(x).unwrap()),
            FieldData::U32(arr) => arr.slice(s![row_idx, ..]).mapv(|x| A::from(x).unwrap()),
            FieldData::U64(arr) => arr.slice(s![row_idx, ..]).mapv(|x| A::from(x).unwrap()),
            FieldData::I8(arr) => arr.slice(s![row_idx, ..]).mapv(|x| A::from(x).unwrap()),
            FieldData::I16(arr) => arr.slice(s![row_idx, ..]).mapv(|x| A::from(x).unwrap()),
            FieldData::I32(arr) => arr.slice(s![row_idx, ..]).mapv(|x| A::from(x).unwrap()),
            FieldData::I64(arr) => arr.slice(s![row_idx, ..]).mapv(|x| A::from(x).unwrap()),
            FieldData::F32(arr) => arr.slice(s![row_idx, ..]).mapv(|x| A::from(x).unwrap()),
            FieldData::F64(arr) => arr.slice(s![row_idx, ..]).mapv(|x| A::from(x).unwrap()),
        }
    }

    /// Return a NumPy array of the specified type.
    pub fn into_pyarray<'py, T: NumpyElement>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<T>>> {
        match self {
            FieldData::U8(arr) => Ok(PyArray2::from_array(py, &arr.mapv(|x| T::from(x).unwrap()))),
            FieldData::U16(arr) => Ok(PyArray2::from_array(py, &arr.mapv(|x| T::from(x).unwrap()))),
            FieldData::U32(arr) => Ok(PyArray2::from_array(py, &arr.mapv(|x| T::from(x).unwrap()))),
            FieldData::U64(arr) => Ok(PyArray2::from_array(py, &arr.mapv(|x| T::from(x).unwrap()))),
            FieldData::I8(arr) => Ok(PyArray2::from_array(py, &arr.mapv(|x| T::from(x).unwrap()))),
            FieldData::I16(arr) => Ok(PyArray2::from_array(py, &arr.mapv(|x| T::from(x).unwrap()))),
            FieldData::I32(arr) => Ok(PyArray2::from_array(py, &arr.mapv(|x| T::from(x).unwrap()))),
            FieldData::I64(arr) => Ok(PyArray2::from_array(py, &arr.mapv(|x| T::from(x).unwrap()))),
            FieldData::F32(arr) => Ok(PyArray2::from_array(py, &arr.mapv(|x| T::from(x).unwrap()))),
            FieldData::F64(arr) => Ok(PyArray2::from_array(py, &arr.mapv(|x| T::from(x).unwrap()))),
        }
    }

    pub fn into_pyarray_shaped<'py, T: NumpyElement>(&self, py: Python<'py>, width: usize, height: usize) -> PyResult<Bound<'py, PyArray3<T>>> {
        assert_eq!(self.npoints(), width * height, "Shape must match number of points");

        match self {
            FieldData::U8(arr) => Ok(PyArray3::from_array(py, &arr.mapv(|x| T::from(x).unwrap()).into_shape_with_order((width, height, self.count())).unwrap())),
            FieldData::U16(arr) => Ok(PyArray3::from_array(py, &arr.mapv(|x| T::from(x).unwrap()).into_shape_with_order((width, height, self.count())).unwrap())),
            FieldData::U32(arr) => Ok(PyArray3::from_array(py, &arr.mapv(|x| T::from(x).unwrap()).into_shape_with_order((width, height, self.count())).unwrap())),
            FieldData::U64(arr) => Ok(PyArray3::from_array(py, &arr.mapv(|x| T::from(x).unwrap()).into_shape_with_order((width, height, self.count())).unwrap())),
            FieldData::I8(arr) => Ok(PyArray3::from_array(py, &arr.mapv(|x| T::from(x).unwrap()).into_shape_with_order((width, height, self.count())).unwrap())),
            FieldData::I16(arr) => Ok(PyArray3::from_array(py, &arr.mapv(|x| T::from(x).unwrap()).into_shape_with_order((width, height, self.count())).unwrap())),
            FieldData::I32(arr) => Ok(PyArray3::from_array(py, &arr.mapv(|x| T::from(x).unwrap()).into_shape_with_order((width, height, self.count())).unwrap())),
            FieldData::I64(arr) => Ok(PyArray3::from_array(py, &arr.mapv(|x| T::from(x).unwrap()).into_shape_with_order((width, height, self.count())).unwrap())),
            FieldData::F32(arr) => Ok(PyArray3::from_array(py, &arr.mapv(|x| T::from(x).unwrap()).into_shape_with_order((width, height, self.count())).unwrap())),
            FieldData::F64(arr) => Ok(PyArray3::from_array(py, &arr.mapv(|x| T::from(x).unwrap()).into_shape_with_order((width, height, self.count())).unwrap())),
        }
    }

    /// Return a sliced version of this field's data (e.g., slice by range).
    /// For example, `[0..5]` for the first 5 points.
    pub fn slice(&self, start: usize, stop: usize, step: usize) -> Self {
        match self {
            FieldData::U8(arr) => FieldData::U8(arr.slice(s![start..stop;step, ..]).to_owned()),
            FieldData::U16(arr) => FieldData::U16(arr.slice(s![start..stop;step, ..]).to_owned()),
            FieldData::U32(arr) => FieldData::U32(arr.slice(s![start..stop;step, ..]).to_owned()),
            FieldData::U64(arr) => FieldData::U64(arr.slice(s![start..stop;step, ..]).to_owned()),
            FieldData::I8(arr) => FieldData::I8(arr.slice(s![start..stop;step, ..]).to_owned()),
            FieldData::I16(arr) => FieldData::I16(arr.slice(s![start..stop;step, ..]).to_owned()),
            FieldData::I32(arr) => FieldData::I32(arr.slice(s![start..stop;step, ..]).to_owned()),
            FieldData::I64(arr) => FieldData::I64(arr.slice(s![start..stop;step, ..]).to_owned()),
            FieldData::F32(arr) => FieldData::F32(arr.slice(s![start..stop;step, ..]).to_owned()),
            FieldData::F64(arr) => FieldData::F64(arr.slice(s![start..stop;step, ..]).to_owned()),
        }
    }

    /// Assign a single row of data to this field.
    pub fn assign_row<A>(&mut self, row_idx: usize, data: &Array1<A>)
    where
        A: Data + NumCast,
    {
        assert_eq!(self.count(), data.len(), "Data length does not match field count");
        assert_eq!(A::DTYPE, self.dtype(), "Expected data type {}, got {}", self.dtype(), A::DTYPE);

        match self {
            FieldData::U8(arr) => {
                for (col, &value) in data.iter().enumerate() {
                    arr[[row_idx, col]] = NumCast::from(value).unwrap();
                }
            }
            FieldData::U16(arr) => {
                for (col, &value) in data.iter().enumerate() {
                    arr[[row_idx, col]] = NumCast::from(value).unwrap();
                }
            }
            FieldData::U32(arr) => {
                for (col, &value) in data.iter().enumerate() {
                    arr[[row_idx, col]] = NumCast::from(value).unwrap();
                }
            }
            FieldData::U64(arr) => {
                for (col, &value) in data.iter().enumerate() {
                    arr[[row_idx, col]] = NumCast::from(value).unwrap();
                }
            }
            FieldData::I8(arr) => {
                for (col, &value) in data.iter().enumerate() {
                    arr[[row_idx, col]] = NumCast::from(value).unwrap();
                }
            }
            FieldData::I16(arr) => {
                for (col, &value) in data.iter().enumerate() {
                    arr[[row_idx, col]] = NumCast::from(value).unwrap();
                }
            }
            FieldData::I32(arr) => {
                for (col, &value) in data.iter().enumerate() {
                    arr[[row_idx, col]] = NumCast::from(value).unwrap();
                }
            }
            FieldData::I64(arr) => {
                for (col, &value) in data.iter().enumerate() {
                    arr[[row_idx, col]] = NumCast::from(value).unwrap();
                }
            }
            FieldData::F32(arr) => {
                for (col, &value) in data.iter().enumerate() {
                    arr[[row_idx, col]] = NumCast::from(value).unwrap();
                }
            }
            FieldData::F64(arr) => {
                for (col, &value) in data.iter().enumerate() {
                    arr[[row_idx, col]] = NumCast::from(value).unwrap();
                }
            }
        }
    }

    /// Assign data from a buffer to this field.
    pub fn assign_from_buffer(&mut self, buffer: &[u8]) {
        let dsize = self.dtype().get_size();
        assert_eq!(buffer.len(), self.len() * dsize, "Buffer length mismatch");

        match self {
            FieldData::U8(arr) => {
                arr.as_slice_mut().unwrap().copy_from_slice(buffer);
            }
            FieldData::U16(arr) => {
                for (i, chunk) in buffer.chunks_exact(dsize).enumerate() {
                    arr.as_slice_mut().unwrap()[i] = u16::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            FieldData::U32(arr) => {
                for (i, chunk) in buffer.chunks_exact(dsize).enumerate() {
                    arr.as_slice_mut().unwrap()[i] = u32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            FieldData::U64(arr) => {
                for (i, chunk) in buffer.chunks_exact(dsize).enumerate() {
                    arr.as_slice_mut().unwrap()[i] = u64::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            FieldData::I8(arr) => {
                for (i, &b) in buffer.iter().enumerate() {
                    arr.as_slice_mut().unwrap()[i] = b as i8;
                }
            }
            FieldData::I16(arr) => {
                for (i, chunk) in buffer.chunks_exact(dsize).enumerate() {
                    arr.as_slice_mut().unwrap()[i] = i16::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            FieldData::I32(arr) => {
                for (i, chunk) in buffer.chunks_exact(dsize).enumerate() {
                    arr.as_slice_mut().unwrap()[i] = i32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            FieldData::I64(arr) => {
                for (i, chunk) in buffer.chunks_exact(dsize).enumerate() {
                    arr.as_slice_mut().unwrap()[i] = i64::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            FieldData::F32(arr) => {
                for (i, chunk) in buffer.chunks_exact(dsize).enumerate() {
                    arr.as_slice_mut().unwrap()[i] = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            FieldData::F64(arr) => {
                for (i, chunk) in buffer.chunks_exact(dsize).enumerate() {
                    arr.as_slice_mut().unwrap()[i] = f64::from_le_bytes(chunk.try_into().unwrap());
                }
            }
        }
    }
}

impl<'py> IntoPyObject<'py> for &FieldData {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            FieldData::U8(arr) => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
            FieldData::U16(arr) => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
            FieldData::U32(arr) => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
            FieldData::U64(arr) => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
            FieldData::I8(arr) => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
            FieldData::I16(arr) => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
            FieldData::I32(arr) => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
            FieldData::I64(arr) => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
            FieldData::F32(arr) => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
            FieldData::F64(arr) => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
        }
    }
}

impl<'py> IntoPyObjectShaped<'py> for &FieldData {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject_shaped(self, py: Python<'py>, width: usize, height: usize) 
        -> PyResult<Self::Output> {
        assert_eq!(self.npoints(), width * height, "Shape must match number of points");

        match self {
            FieldData::U8(arr) => Ok(PyArray3::from_array(py, &arr.clone().into_shape_with_order((width, height, self.count())).unwrap()).into_bound_py_any(py)?),
            FieldData::U16(arr) => Ok(PyArray3::from_array(py, &arr.clone().into_shape_with_order((width, height, self.count())).unwrap()).into_bound_py_any(py)?),
            FieldData::U32(arr) => Ok(PyArray3::from_array(py, &arr.clone().into_shape_with_order((width, height, self.count())).unwrap()).into_bound_py_any(py)?),
            FieldData::U64(arr) => Ok(PyArray3::from_array(py, &arr.clone().into_shape_with_order((width, height, self.count())).unwrap()).into_bound_py_any(py)?),
            FieldData::I8(arr) => Ok(PyArray3::from_array(py, &arr.clone().into_shape_with_order((width, height, self.count())).unwrap()).into_bound_py_any(py)?),
            FieldData::I16(arr) => Ok(PyArray3::from_array(py, &arr.clone().into_shape_with_order((width, height, self.count())).unwrap()).into_bound_py_any(py)?),
            FieldData::I32(arr) => Ok(PyArray3::from_array(py, &arr.clone().into_shape_with_order((width, height, self.count())).unwrap()).into_bound_py_any(py)?),
            FieldData::I64(arr) => Ok(PyArray3::from_array(py, &arr.clone().into_shape_with_order((width, height, self.count())).unwrap()).into_bound_py_any(py)?),
            FieldData::F32(arr) => Ok(PyArray3::from_array(py, &arr.clone().into_shape_with_order((width, height, self.count())).unwrap()).into_bound_py_any(py)?),
            FieldData::F64(arr) => Ok(PyArray3::from_array(py, &arr.clone().into_shape_with_order((width, height, self.count())).unwrap()).into_bound_py_any(py)?),
        }
    }
}

impl std::fmt::Display for FieldData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldData::U8(arr) => write!(f, "{:}", arr),
            FieldData::U16(arr) => write!(f, "{:}", arr),
            FieldData::U32(arr) => write!(f, "{:}", arr),
            FieldData::U64(arr) => write!(f, "{:}", arr),
            FieldData::I8(arr) => write!(f, "{:}", arr),
            FieldData::I16(arr) => write!(f, "{:}", arr),
            FieldData::I32(arr) => write!(f, "{:}", arr),
            FieldData::I64(arr) => write!(f, "{:}", arr),
            FieldData::F32(arr) => write!(f, "{:}", arr),
            FieldData::F64(arr) => write!(f, "{:}", arr),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple () {
        let arr = Array2::from(vec![[1], [2], [3], [4], [5]]);
        let field = FieldData::U8(arr);
        assert_eq!(field.npoints(), 5);
        assert_eq!(field.dtype().get_size(), 1);
        assert_eq!(field.dtype().get_type(), "U");
    }

    #[test]
    fn test_slicing () {
        let arr = Array2::from(vec![[1], [2], [3], [4], [5]]);
        let field = FieldData::U8(arr);
        let sliced = field.slice(1, 4, 1);
        assert_eq!(sliced.npoints(), 3);
        assert_eq!(sliced.dtype().get_size(), 1);
        assert_eq!(sliced.dtype().get_type(), "U");

        let arr = Array2::from(vec![[1], [2], [3], [4], [5]]);
        let field = FieldData::U8(arr);
        let sliced = field.slice(0, 5, 2);
        assert_eq!(sliced.npoints(), 3);
        assert_eq!(sliced.dtype().get_size(), 1);
        assert_eq!(sliced.dtype().get_type(), "U");
    }

    #[test]
    fn test_construction () {
        let mut field = FieldData::new(Dtype::U8, 10, 3);
        assert_eq!(field.npoints(), 10);

        let data = Array1::from(vec![1, 2, 3]);
        field.assign_row(0, &data);
        assert_eq!(field.npoints(), 10);
        assert_eq!(field.get_row::<u8>(0), data);
    }
}
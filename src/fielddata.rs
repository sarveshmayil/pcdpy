use num_traits::NumCast;
use pyo3::{exceptions::PyValueError, prelude::*, BoundObject, IntoPyObject, IntoPyObjectExt};
use ndarray::{Array1, Array2, s};
use numpy::{PyArray2, PyArray3, Element, PyReadonlyArray2};
use crate::metadata::{Data, Dtype};

/// A trait for elements that can be used in numpy conversions.
pub trait NumpyElement: Element + NumCast {}
impl<T: Element + NumCast> NumpyElement for T {}

/// A trait for converting an object into a shaped Python object.
pub trait IntoPyObjectShaped<'py> {
    type Target;
    type Output;
    type Error;

    fn into_pyobject_shaped(self, py: Python<'py>, width: usize, height: usize) 
        -> Result<Self::Output, Self::Error>;
}

// =====================================================================
// Helper Macros for reducing code duplication across variants
// =====================================================================

macro_rules! match_get_data {
    ($self:expr, $target:ty) => {
         match $self {
             FieldData::U8(arr)  => arr.mapv(|x| <$target>::from(x).unwrap()),
             FieldData::U16(arr) => arr.mapv(|x| <$target>::from(x).unwrap()),
             FieldData::U32(arr) => arr.mapv(|x| <$target>::from(x).unwrap()),
             FieldData::U64(arr) => arr.mapv(|x| <$target>::from(x).unwrap()),
             FieldData::I8(arr)  => arr.mapv(|x| <$target>::from(x).unwrap()),
             FieldData::I16(arr) => arr.mapv(|x| <$target>::from(x).unwrap()),
             FieldData::I32(arr) => arr.mapv(|x| <$target>::from(x).unwrap()),
             FieldData::I64(arr) => arr.mapv(|x| <$target>::from(x).unwrap()),
             FieldData::F32(arr) => arr.mapv(|x| <$target>::from(x).unwrap()),
             FieldData::F64(arr) => arr.mapv(|x| <$target>::from(x).unwrap()),
         }
    }
}

macro_rules! match_get_row {
    ($self:expr, $row_idx:expr, $target:ty) => {
         match $self {
             FieldData::U8(arr)  => arr.slice(s![$row_idx, ..]).mapv(|x| <$target>::from(x).unwrap()),
             FieldData::U16(arr) => arr.slice(s![$row_idx, ..]).mapv(|x| <$target>::from(x).unwrap()),
             FieldData::U32(arr) => arr.slice(s![$row_idx, ..]).mapv(|x| <$target>::from(x).unwrap()),
             FieldData::U64(arr) => arr.slice(s![$row_idx, ..]).mapv(|x| <$target>::from(x).unwrap()),
             FieldData::I8(arr)  => arr.slice(s![$row_idx, ..]).mapv(|x| <$target>::from(x).unwrap()),
             FieldData::I16(arr) => arr.slice(s![$row_idx, ..]).mapv(|x| <$target>::from(x).unwrap()),
             FieldData::I32(arr) => arr.slice(s![$row_idx, ..]).mapv(|x| <$target>::from(x).unwrap()),
             FieldData::I64(arr) => arr.slice(s![$row_idx, ..]).mapv(|x| <$target>::from(x).unwrap()),
             FieldData::F32(arr) => arr.slice(s![$row_idx, ..]).mapv(|x| <$target>::from(x).unwrap()),
             FieldData::F64(arr) => arr.slice(s![$row_idx, ..]).mapv(|x| <$target>::from(x).unwrap()),
         }
    }
}

macro_rules! match_slice {
    ($self:expr, $start:expr, $stop:expr, $step:expr) => {
         match $self {
             FieldData::U8(arr)  => FieldData::U8(arr.slice(s![$start..$stop;$step, ..]).to_owned()),
             FieldData::U16(arr) => FieldData::U16(arr.slice(s![$start..$stop;$step, ..]).to_owned()),
             FieldData::U32(arr) => FieldData::U32(arr.slice(s![$start..$stop;$step, ..]).to_owned()),
             FieldData::U64(arr) => FieldData::U64(arr.slice(s![$start..$stop;$step, ..]).to_owned()),
             FieldData::I8(arr)  => FieldData::I8(arr.slice(s![$start..$stop;$step, ..]).to_owned()),
             FieldData::I16(arr) => FieldData::I16(arr.slice(s![$start..$stop;$step, ..]).to_owned()),
             FieldData::I32(arr) => FieldData::I32(arr.slice(s![$start..$stop;$step, ..]).to_owned()),
             FieldData::I64(arr) => FieldData::I64(arr.slice(s![$start..$stop;$step, ..]).to_owned()),
             FieldData::F32(arr) => FieldData::F32(arr.slice(s![$start..$stop;$step, ..]).to_owned()),
             FieldData::F64(arr) => FieldData::F64(arr.slice(s![$start..$stop;$step, ..]).to_owned()),
         }
    }
}

macro_rules! match_assign_row {
    ($self:expr, $row_idx:expr, $data:expr) => {
         match $self {
             FieldData::U8(arr) => {
                 for (col, &value) in $data.iter().enumerate() {
                     arr[[$row_idx, col]] = NumCast::from(value).unwrap();
                 }
             },
             FieldData::U16(arr) => {
                 for (col, &value) in $data.iter().enumerate() {
                     arr[[$row_idx, col]] = NumCast::from(value).unwrap();
                 }
             },
             FieldData::U32(arr) => {
                 for (col, &value) in $data.iter().enumerate() {
                     arr[[$row_idx, col]] = NumCast::from(value).unwrap();
                 }
             },
             FieldData::U64(arr) => {
                 for (col, &value) in $data.iter().enumerate() {
                     arr[[$row_idx, col]] = NumCast::from(value).unwrap();
                 }
             },
             FieldData::I8(arr) => {
                 for (col, &value) in $data.iter().enumerate() {
                     arr[[$row_idx, col]] = NumCast::from(value).unwrap();
                 }
             },
             FieldData::I16(arr) => {
                 for (col, &value) in $data.iter().enumerate() {
                     arr[[$row_idx, col]] = NumCast::from(value).unwrap();
                 }
             },
             FieldData::I32(arr) => {
                 for (col, &value) in $data.iter().enumerate() {
                     arr[[$row_idx, col]] = NumCast::from(value).unwrap();
                 }
             },
             FieldData::I64(arr) => {
                 for (col, &value) in $data.iter().enumerate() {
                     arr[[$row_idx, col]] = NumCast::from(value).unwrap();
                 }
             },
             FieldData::F32(arr) => {
                 for (col, &value) in $data.iter().enumerate() {
                     arr[[$row_idx, col]] = NumCast::from(value).unwrap();
                 }
             },
             FieldData::F64(arr) => {
                 for (col, &value) in $data.iter().enumerate() {
                     arr[[$row_idx, col]] = NumCast::from(value).unwrap();
                 }
             },
         }
    }
}

macro_rules! match_assign_from_buffer {
    ($self:expr, $buffer:expr) => {{
         let dsize = $self.dtype().get_size();
         assert_eq!($buffer.len(), $self.len() * dsize, "Buffer length mismatch");
         match $self {
             FieldData::U8(arr) => {
                 arr.as_slice_mut().unwrap().copy_from_slice($buffer);
             },
             FieldData::U16(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.as_slice_mut().unwrap()[i] = u16::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::U32(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.as_slice_mut().unwrap()[i] = u32::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::U64(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.as_slice_mut().unwrap()[i] = u64::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::I8(arr) => {
                 for (i, &b) in $buffer.iter().enumerate() {
                     arr.as_slice_mut().unwrap()[i] = b as i8;
                 }
             },
             FieldData::I16(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.as_slice_mut().unwrap()[i] = i16::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::I32(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.as_slice_mut().unwrap()[i] = i32::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::I64(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.as_slice_mut().unwrap()[i] = i64::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::F32(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.as_slice_mut().unwrap()[i] = f32::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::F64(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.as_slice_mut().unwrap()[i] = f64::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
         }
    }}
}

macro_rules! match_assign_row_from_buffer {
    ($self:expr, $row_idx:expr, $buffer:expr) => {{
         let dsize = $self.dtype().get_size();
         assert_eq!($buffer.len(), $self.count() * dsize, "Buffer length mismatch");
         match $self {
             FieldData::U8(arr) => {
                 arr.slice_mut(s![$row_idx, ..]).as_slice_mut().unwrap().copy_from_slice($buffer);
             },
             FieldData::U16(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.slice_mut(s![$row_idx, ..]).as_slice_mut().unwrap()[i] = u16::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::U32(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.slice_mut(s![$row_idx, ..]).as_slice_mut().unwrap()[i] = u32::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::U64(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.slice_mut(s![$row_idx, ..]).as_slice_mut().unwrap()[i] = u64::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::I8(arr) => {
                 for (i, &b) in $buffer.iter().enumerate() {
                     arr.slice_mut(s![$row_idx, ..]).as_slice_mut().unwrap()[i] = b as i8;
                 }
             },
             FieldData::I16(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.slice_mut(s![$row_idx, ..]).as_slice_mut().unwrap()[i] = i16::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::I32(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.slice_mut(s![$row_idx, ..]).as_slice_mut().unwrap()[i] = i32::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::I64(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.slice_mut(s![$row_idx, ..]).as_slice_mut().unwrap()[i] = i64::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::F32(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.slice_mut(s![$row_idx, ..]).as_slice_mut().unwrap()[i] = f32::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
             FieldData::F64(arr) => {
                 for (i, chunk) in $buffer.chunks_exact(dsize).enumerate() {
                     arr.slice_mut(s![$row_idx, ..]).as_slice_mut().unwrap()[i] = f64::from_le_bytes(chunk.try_into().unwrap());
                 }
             },
         }
    }}
}

// =====================================================================
// FieldData Implementation
// =====================================================================

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
            Dtype::U8  => FieldData::U8(Array2::zeros((npoints, count))),
            Dtype::U16 => FieldData::U16(Array2::zeros((npoints, count))),
            Dtype::U32 => FieldData::U32(Array2::zeros((npoints, count))),
            Dtype::U64 => FieldData::U64(Array2::zeros((npoints, count))),
            Dtype::I8  => FieldData::I8(Array2::zeros((npoints, count))),
            Dtype::I16 => FieldData::I16(Array2::zeros((npoints, count))),
            Dtype::I32 => FieldData::I32(Array2::zeros((npoints, count))),
            Dtype::I64 => FieldData::I64(Array2::zeros((npoints, count))),
            Dtype::F32 => FieldData::F32(Array2::zeros((npoints, count))),
            Dtype::F64 => FieldData::F64(Array2::zeros((npoints, count))),
        }
    }

    pub fn from_pyarray<'py>(pyarray: &Bound<'py, PyAny>, dtype: Dtype) -> PyResult<Self> {
        match dtype {
            Dtype::U8 => Ok(FieldData::U8(pyarray.extract::<PyReadonlyArray2<u8>>()?.as_array().to_owned())),
            Dtype::U16 => Ok(FieldData::U16(pyarray.extract::<PyReadonlyArray2<u16>>()?.as_array().to_owned())),
            Dtype::U32 => Ok(FieldData::U32(pyarray.extract::<PyReadonlyArray2<u32>>()?.as_array().to_owned())),
            Dtype::U64 => Ok(FieldData::U64(pyarray.extract::<PyReadonlyArray2<u64>>()?.as_array().to_owned())),
            Dtype::I8 => Ok(FieldData::I8(pyarray.extract::<PyReadonlyArray2<i8>>()?.as_array().to_owned())),
            Dtype::I16 => Ok(FieldData::I16(pyarray.extract::<PyReadonlyArray2<i16>>()?.as_array().to_owned())),
            Dtype::I32 => Ok(FieldData::I32(pyarray.extract::<PyReadonlyArray2<i32>>()?.as_array().to_owned())),
            Dtype::I64 => Ok(FieldData::I64(pyarray.extract::<PyReadonlyArray2<i64>>()?.as_array().to_owned())),
            Dtype::F32 => Ok(FieldData::F32(pyarray.extract::<PyReadonlyArray2<f32>>()?.as_array().to_owned())),
            Dtype::F64 => Ok(FieldData::F64(pyarray.extract::<PyReadonlyArray2<f64>>()?.as_array().to_owned())),
        }
    }

    /// Return the length (total number of values) in this field.
    pub fn len(&self) -> usize {
        match self {
            FieldData::U8(arr)   => arr.len(),
            FieldData::U16(arr) => arr.len(),
            FieldData::U32(arr) => arr.len(),
            FieldData::U64(arr) => arr.len(),
            FieldData::I8(arr)   => arr.len(),
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
            FieldData::U8(arr)   => arr.shape()[0],
            FieldData::U16(arr) => arr.shape()[0],
            FieldData::U32(arr) => arr.shape()[0],
            FieldData::U64(arr) => arr.shape()[0],
            FieldData::I8(arr)   => arr.shape()[0],
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
            FieldData::U8(arr)   => arr.shape()[1],
            FieldData::U16(arr) => arr.shape()[1],
            FieldData::U32(arr) => arr.shape()[1],
            FieldData::U64(arr) => arr.shape()[1],
            FieldData::I8(arr)   => arr.shape()[1],
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
            FieldData::U8(_)  => Dtype::U8,
            FieldData::U16(_) => Dtype::U16,
            FieldData::U32(_) => Dtype::U32,
            FieldData::U64(_) => Dtype::U64,
            FieldData::I8(_)  => Dtype::I8,
            FieldData::I16(_) => Dtype::I16,
            FieldData::I32(_) => Dtype::I32,
            FieldData::I64(_) => Dtype::I64,
            FieldData::F32(_) => Dtype::F32,
            FieldData::F64(_) => Dtype::F64,
        }
    }

    /// Return the data as a 2D array of the specified type.
    pub fn get_data<A: Data + NumCast>(&self) -> Array2<A> {
        match_get_data!(self, A)
    }

    /// Return a single row of data as a 1D array of the specified type.
    pub fn get_row<A: Data + NumCast>(&self, row_idx: usize) -> Array1<A> {
        match_get_row!(self, row_idx, A)
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

    /// Return a NumPy array reshaped into (width, height, count).
    pub fn into_pyarray_shaped<'py, T: NumpyElement>(&self, py: Python<'py>, width: usize, height: usize) -> PyResult<Bound<'py, PyArray3<T>>> {
        if self.npoints() != width * height {
            return Err(PyValueError::new_err("Shape must match number of points"));
        }

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
    pub fn slice(&self, start: usize, stop: usize, step: usize) -> Self {
        match_slice!(self, start, stop, step)
    }

    /// Assign a single row of data to this field.
    pub fn assign_row<A>(&mut self, row_idx: usize, data: &Array1<A>)
    where
        A: Data + NumCast,
    {
        assert_eq!(self.count(), data.len(), "Data length does not match field count");
        assert_eq!(A::DTYPE, self.dtype(), "Expected data type {}, got {}", self.dtype(), A::DTYPE);
        match_assign_row!(self, row_idx, data);
    }

    /// Assign data from a buffer to this field.
    pub fn assign_from_buffer(&mut self, buffer: &[u8]) {
        match_assign_from_buffer!(self, buffer);
    }

    /// Assign a single row of data from a buffer to this field.
    pub fn assign_row_from_buffer(&mut self, row_idx: usize, buffer: &[u8]) {
        match_assign_row_from_buffer!(self, row_idx, buffer);
    }

    /// Update a strided slice of self with a strided slice from new_field.
    ///
    /// - `orig_range`: The range of row indices in self to update.
    /// - `orig_step`: The step (stride) for the rows in self.
    /// - `new_range`: The range of row indices in new_field to copy from.
    /// - `new_step`: The step (stride) for the rows in new_field.
    ///
    /// Returns an error if the number of rows in both slices do not match
    /// or if the two FieldData variants differ.
    pub fn update_slice_strided(
        &mut self,
        new_field: &FieldData,
        orig_range: std::ops::Range<usize>,
        orig_step: usize,
        new_range: std::ops::Range<usize>,
        new_step: usize,
    ) -> PyResult<()> {
        // Calculate the number of rows in each slice.
        let num_orig_rows = (orig_range.end.saturating_sub(orig_range.start) + orig_step - 1) / orig_step;
        let num_new_rows = (new_range.end.saturating_sub(new_range.start) + new_step - 1) / new_step;
        if num_orig_rows != num_new_rows {
            return Err(PyValueError::new_err("Slice lengths do not match"));
        }
        // Create slicing specifications for both arrays.
        let orig_slice = s![orig_range.start..orig_range.end; orig_step, ..];
        let new_slice = s![new_range.start..new_range.end; new_step, ..];

        // Use ndarray's assign method to update the slice.
        match (self, new_field) {
            (FieldData::U8(ref mut orig_arr), FieldData::U8(new_arr)) => {
                orig_arr.slice_mut(orig_slice).assign(&new_arr.slice(new_slice));
            },
            (FieldData::U16(ref mut orig_arr), FieldData::U16(new_arr)) => {
                orig_arr.slice_mut(orig_slice).assign(&new_arr.slice(new_slice));
            },
            (FieldData::U32(ref mut orig_arr), FieldData::U32(new_arr)) => {
                orig_arr.slice_mut(orig_slice).assign(&new_arr.slice(new_slice));
            },
            (FieldData::U64(ref mut orig_arr), FieldData::U64(new_arr)) => {
                orig_arr.slice_mut(orig_slice).assign(&new_arr.slice(new_slice));
            },
            (FieldData::I8(ref mut orig_arr), FieldData::I8(new_arr)) => {
                orig_arr.slice_mut(orig_slice).assign(&new_arr.slice(new_slice));
            },
            (FieldData::I16(ref mut orig_arr), FieldData::I16(new_arr)) => {
                orig_arr.slice_mut(orig_slice).assign(&new_arr.slice(new_slice));
            },
            (FieldData::I32(ref mut orig_arr), FieldData::I32(new_arr)) => {
                orig_arr.slice_mut(orig_slice).assign(&new_arr.slice(new_slice));
            },
            (FieldData::I64(ref mut orig_arr), FieldData::I64(new_arr)) => {
                orig_arr.slice_mut(orig_slice).assign(&new_arr.slice(new_slice));
            },
            (FieldData::F32(ref mut orig_arr), FieldData::F32(new_arr)) => {
                orig_arr.slice_mut(orig_slice).assign(&new_arr.slice(new_slice));
            },
            (FieldData::F64(ref mut orig_arr), FieldData::F64(new_arr)) => {
                orig_arr.slice_mut(orig_slice).assign(&new_arr.slice(new_slice));
            },
            _ => return Err(PyValueError::new_err("Field types do not match for slice assignment")),
        }
        Ok(())
    }
}

impl<'py> IntoPyObject<'py> for &FieldData {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            FieldData::U8(arr)   => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
            FieldData::U16(arr) => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
            FieldData::U32(arr) => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
            FieldData::U64(arr) => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
            FieldData::I8(arr)   => Ok(PyArray2::from_array(py, arr).into_bound_py_any(py)?),
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

    fn into_pyobject_shaped(self, py: Python<'py>, width: usize, height: usize) -> PyResult<Self::Output> {
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
            FieldData::U8(arr)   => write!(f, "{}", arr),
            FieldData::U16(arr) => write!(f, "{}", arr),
            FieldData::U32(arr) => write!(f, "{}", arr),
            FieldData::U64(arr) => write!(f, "{}", arr),
            FieldData::I8(arr)   => write!(f, "{}", arr),
            FieldData::I16(arr) => write!(f, "{}", arr),
            FieldData::I32(arr) => write!(f, "{}", arr),
            FieldData::I64(arr) => write!(f, "{}", arr),
            FieldData::F32(arr) => write!(f, "{}", arr),
            FieldData::F64(arr) => write!(f, "{}", arr),
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
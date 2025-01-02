use ndarray::{Array1, s};
use crate::metadata::Dtype;

#[derive(Debug, Clone, PartialEq)]
pub enum FieldData {
    U8(Array1<u8>),
    U16(Array1<u16>),
    U32(Array1<u32>),
    U64(Array1<u64>),
    I8(Array1<i8>),
    I16(Array1<i16>),
    I32(Array1<i32>),
    I64(Array1<i64>),
    F32(Array1<f32>),
    F64(Array1<f64>),
}

impl FieldData {
    /// Return the length (number of points) in this field.
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

    /// Return a sliced version of this field's data (e.g., slice by range).
    /// For example, `[0..5]` for the first 5 points.
    pub fn slice(&self, start: usize, stop: usize, step: usize) -> Self {
        match self {
            FieldData::U8(arr) => FieldData::U8(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::U16(arr) => FieldData::U16(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::U32(arr) => FieldData::U32(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::U64(arr) => FieldData::U64(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::I8(arr) => FieldData::I8(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::I16(arr) => FieldData::I16(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::I32(arr) => FieldData::I32(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::I64(arr) => FieldData::I64(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::F32(arr) => FieldData::F32(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::F64(arr) => FieldData::F64(arr.slice(s![start..stop;step]).to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u8 () {
        let arr = Array1::from(vec![1, 2, 3, 4, 5]);
        let field = FieldData::U8(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.dtype().get_size(), 1);
        assert_eq!(field.dtype().get_type(), "U");
        assert_eq!(field.slice(0, 3, 1), FieldData::U8(Array1::from(vec![1, 2, 3])));
    }

    #[test]
    fn test_u16 () {
        let arr = Array1::from(vec![1, 2, 3, 4, 5]);
        let field = FieldData::U16(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.dtype().get_size(), 2);
        assert_eq!(field.dtype().get_type(), "U");
        assert_eq!(field.slice(0, 3, 1), FieldData::U16(Array1::from(vec![1, 2, 3])));
    }

    #[test]
    fn test_u32 () {
        let arr = Array1::from(vec![1, 2, 3, 4, 5]);
        let field = FieldData::U32(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.dtype().get_size(), 4);
        assert_eq!(field.dtype().get_type(), "U");
        assert_eq!(field.slice(0, 3, 1), FieldData::U32(Array1::from(vec![1, 2, 3])));
    }

    #[test]
    fn test_u64 () {
        let arr = Array1::from(vec![1, 2, 3, 4, 5]);
        let field = FieldData::U64(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.dtype().get_size(), 8);
        assert_eq!(field.dtype().get_type(), "U");
        assert_eq!(field.slice(0, 3, 1), FieldData::U64(Array1::from(vec![1, 2, 3])));
    }

    #[test]
    fn test_i8 () {
        let arr = Array1::from(vec![-1, -2, -3, -4, -5]);
        let field = FieldData::I8(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.dtype().get_size(), 1);
        assert_eq!(field.dtype().get_type(), "I");
        assert_eq!(field.slice(0, 3, 1), FieldData::I8(Array1::from(vec![-1, -2, -3])));
    }

    #[test]
    fn test_i16 () {
        let arr = Array1::from(vec![-1, -2, -3, -4, -5]);
        let field = FieldData::I16(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.dtype().get_size(), 2);
        assert_eq!(field.dtype().get_type(), "I");
        assert_eq!(field.slice(0, 3, 1), FieldData::I16(Array1::from(vec![-1, -2, -3])));
    }

    #[test]
    fn test_i32 () {
        let arr = Array1::from(vec![-1, -2, -3, -4, -5]);
        let field = FieldData::I32(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.dtype().get_size(), 4);
        assert_eq!(field.dtype().get_type(), "I");
        assert_eq!(field.slice(0, 3, 1), FieldData::I32(Array1::from(vec![-1, -2, -3])));
    }

    #[test]
    fn test_i64 () {
        let arr = Array1::from(vec![-1, -2, -3, -4, -5]);
        let field = FieldData::I64(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.dtype().get_size(), 8);
        assert_eq!(field.dtype().get_type(), "I");
        assert_eq!(field.slice(0, 3, 1), FieldData::I64(Array1::from(vec![-1, -2, -3])));
    }

    #[test]
    fn test_f32 () {
        let arr = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let field = FieldData::F32(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.dtype().get_size(), 4);
        assert_eq!(field.dtype().get_type(), "F");
        assert_eq!(field.slice(0, 3, 1), FieldData::F32(Array1::from(vec![1.0, 2.0, 3.0])));
    }

    #[test]
    fn test_f64 () {
        let arr = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let field = FieldData::F64(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.dtype().get_size(), 8);
        assert_eq!(field.dtype().get_type(), "F");
        assert_eq!(field.slice(0, 3, 1), FieldData::F64(Array1::from(vec![1.0, 2.0, 3.0])));
    }
}
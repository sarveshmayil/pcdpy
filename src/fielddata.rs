use ndarray::{Array1, s};

#[derive(Debug, Clone)]
pub enum FieldData {
    U1(Array1<u8>),
    U2(Array1<u16>),
    U4(Array1<u32>),
    U8(Array1<u64>),
    I1(Array1<i8>),
    I2(Array1<i16>),
    I4(Array1<i32>),
    I8(Array1<i64>),
    F4(Array1<f32>),
    F8(Array1<f64>),
}

impl FieldData {
    /// Return the length (number of points) in this field.
    pub fn len(&self) -> usize {
        match self {
            FieldData::U1(arr) => arr.len(),
            FieldData::U2(arr) => arr.len(),
            FieldData::U4(arr) => arr.len(),
            FieldData::U8(arr) => arr.len(),
            FieldData::I1(arr) => arr.len(),
            FieldData::I2(arr) => arr.len(),
            FieldData::I4(arr) => arr.len(),
            FieldData::I8(arr) => arr.len(),
            FieldData::F4(arr) => arr.len(),
            FieldData::F8(arr) => arr.len(),
        }
    }

    /// Return number of bytes for each element in this field
    pub fn get_size(&self) -> usize {
        match self {
            FieldData::U1(_) => 1,
            FieldData::U2(_) => 2,
            FieldData::U4(_) => 4,
            FieldData::U8(_) => 8,
            FieldData::I1(_) => 1,
            FieldData::I2(_) => 2,
            FieldData::I4(_) => 4,
            FieldData::I8(_) => 8,
            FieldData::F4(_) => 4,
            FieldData::F8(_) => 8,
        }
    }

    /// Return field type: 'U' for unsigned int, 'I' for signed int, 'F' for float
    pub fn get_type(&self) -> &str {
        match self {
            FieldData::U1(_) | FieldData::U2(_) | FieldData::U4(_) | FieldData::U8(_) => "U",
            FieldData::I1(_) | FieldData::I2(_) | FieldData::I4(_) | FieldData::I8(_) => "I",
            FieldData::F4(_) | FieldData::F8(_) => "F",
        }
    }

    /// Return a sliced version of this field's data (e.g., slice by range).
    /// For example, `[0..5]` for the first 5 points.
    pub fn slice(&self, start: usize, stop: usize, step: usize) -> Self {
        match self {
            FieldData::U1(arr) => FieldData::U1(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::U2(arr) => FieldData::U2(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::U4(arr) => FieldData::U4(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::U8(arr) => FieldData::U8(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::I1(arr) => FieldData::I1(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::I2(arr) => FieldData::I2(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::I4(arr) => FieldData::I4(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::I8(arr) => FieldData::I8(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::F4(arr) => FieldData::F4(arr.slice(s![start..stop;step]).to_owned()),
            FieldData::F8(arr) => FieldData::F8(arr.slice(s![start..stop;step]).to_owned()),
        }
    }
}

impl PartialEq for FieldData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FieldData::U1(a), FieldData::U1(b)) => a == b,
            (FieldData::U2(a), FieldData::U2(b)) => a == b,
            (FieldData::U4(a), FieldData::U4(b)) => a == b,
            (FieldData::U8(a), FieldData::U8(b)) => a == b,
            (FieldData::I1(a), FieldData::I1(b)) => a == b,
            (FieldData::I2(a), FieldData::I2(b)) => a == b,
            (FieldData::I4(a), FieldData::I4(b)) => a == b,
            (FieldData::I8(a), FieldData::I8(b)) => a == b,
            (FieldData::F4(a), FieldData::F4(b)) => a == b,
            (FieldData::F8(a), FieldData::F8(b)) => a == b,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u1 () {
        let arr = Array1::from(vec![1, 2, 3, 4, 5]);
        let field = FieldData::U1(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.get_size(), 1);
        assert_eq!(field.get_type(), "U");
        assert_eq!(field.slice(0, 3, 1), FieldData::U1(Array1::from(vec![1, 2, 3])));
    }

    #[test]
    fn test_u2 () {
        let arr = Array1::from(vec![1, 2, 3, 4, 5]);
        let field = FieldData::U2(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.get_size(), 2);
        assert_eq!(field.get_type(), "U");
        assert_eq!(field.slice(0, 3, 1), FieldData::U2(Array1::from(vec![1, 2, 3])));
    }

    #[test]
    fn test_u4 () {
        let arr = Array1::from(vec![1, 2, 3, 4, 5]);
        let field = FieldData::U4(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.get_size(), 4);
        assert_eq!(field.get_type(), "U");
        assert_eq!(field.slice(0, 3, 1), FieldData::U4(Array1::from(vec![1, 2, 3])));
    }

    #[test]
    fn test_u8 () {
        let arr = Array1::from(vec![1, 2, 3, 4, 5]);
        let field = FieldData::U8(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.get_size(), 8);
        assert_eq!(field.get_type(), "U");
        assert_eq!(field.slice(0, 3, 1), FieldData::U8(Array1::from(vec![1, 2, 3])));
    }

    #[test]
    fn test_i1 () {
        let arr = Array1::from(vec![-1, -2, -3, -4, -5]);
        let field = FieldData::I1(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.get_size(), 1);
        assert_eq!(field.get_type(), "I");
        assert_eq!(field.slice(0, 3, 1), FieldData::I1(Array1::from(vec![-1, -2, -3])));
    }

    #[test]
    fn test_i2 () {
        let arr = Array1::from(vec![-1, -2, -3, -4, -5]);
        let field = FieldData::I2(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.get_size(), 2);
        assert_eq!(field.get_type(), "I");
        assert_eq!(field.slice(0, 3, 1), FieldData::I2(Array1::from(vec![-1, -2, -3])));
    }

    #[test]
    fn test_i4 () {
        let arr = Array1::from(vec![-1, -2, -3, -4, -5]);
        let field = FieldData::I4(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.get_size(), 4);
        assert_eq!(field.get_type(), "I");
        assert_eq!(field.slice(0, 3, 1), FieldData::I4(Array1::from(vec![-1, -2, -3])));
    }

    #[test]
    fn test_i8 () {
        let arr = Array1::from(vec![-1, -2, -3, -4, -5]);
        let field = FieldData::I8(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.get_size(), 8);
        assert_eq!(field.get_type(), "I");
        assert_eq!(field.slice(0, 3, 1), FieldData::I8(Array1::from(vec![-1, -2, -3])));
    }

    #[test]
    fn test_f4 () {
        let arr = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let field = FieldData::F4(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.get_size(), 4);
        assert_eq!(field.get_type(), "F");
        assert_eq!(field.slice(0, 3, 1), FieldData::F4(Array1::from(vec![1.0, 2.0, 3.0])));
    }

    #[test]
    fn test_f8 () {
        let arr = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let field = FieldData::F8(arr);
        assert_eq!(field.len(), 5);
        assert_eq!(field.get_size(), 8);
        assert_eq!(field.get_type(), "F");
        assert_eq!(field.slice(0, 3, 1), FieldData::F8(Array1::from(vec![1.0, 2.0, 3.0])));
    }
}
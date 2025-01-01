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
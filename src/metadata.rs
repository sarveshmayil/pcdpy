use std::{iter::FromIterator, ops::{Index, IndexMut}};

#[derive(Debug, Clone, PartialEq)]
pub struct Metadata {
    pub fields: FieldSchema,
    pub width: usize,
    pub height: usize,
    pub npoints: usize,
    pub viewpoint: Viewpoint,
    pub encoding: Encoding,
    pub version: String,
}

impl Metadata {
    pub fn new() -> Self {
        Self {
            fields: FieldSchema::new(),
            width: 0,
            height: 1,
            viewpoint: Viewpoint::default(),
            npoints: 0,
            encoding: Encoding::default(),
            version: "0.7".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}
impl Dtype {
    pub fn get_size(&self) -> usize {
        match self {
            Dtype::U8 | Dtype::I8 => 1,
            Dtype::U16 | Dtype::I16 => 2,
            Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
            Dtype::U64 | Dtype::I64 | Dtype::F64 => 8,
        }
    }

    pub fn get_type(&self) -> &str {
        match self {
            Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 => "U",
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 => "I",
            Dtype::F32 | Dtype::F64 => "F",
        }
    }

    pub fn from_type_size(t: &str, s: &usize) -> Self {
        match (t, s) {
            ("U", 1) => Dtype::U8,
            ("U", 2) => Dtype::U16,
            ("U", 4) => Dtype::U32,
            ("U", 8) => Dtype::U64,
            ("I", 1) => Dtype::I8,
            ("I", 2) => Dtype::I16,
            ("I", 4) => Dtype::I32,
            ("I", 8) => Dtype::I64,
            ("F", 4) => Dtype::F32,
            ("F", 8) => Dtype::F64,
            _ => panic!("Field type {} and size {} is not supported", t, s),
        }
    }
}
impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub trait Data 
where Self: Copy 
{
    const DTYPE: Dtype;
}
impl Data for u8 { const DTYPE: Dtype = Dtype::U8; }
impl Data for u16 { const DTYPE: Dtype = Dtype::U16; }
impl Data for u32 { const DTYPE: Dtype = Dtype::U32; }
impl Data for u64 { const DTYPE: Dtype = Dtype::U64; }
impl Data for i8 { const DTYPE: Dtype = Dtype::I8; }
impl Data for i16 { const DTYPE: Dtype = Dtype::I16; }
impl Data for i32 { const DTYPE: Dtype = Dtype::I32; }
impl Data for i64 { const DTYPE: Dtype = Dtype::I64; }
impl Data for f32 { const DTYPE: Dtype = Dtype::F32; }
impl Data for f64 { const DTYPE: Dtype = Dtype::F64; }

#[derive(Debug, Clone, PartialEq)]
pub struct Viewpoint {
    pub tx: f32,
    pub ty: f32,
    pub tz: f32,
    pub qw: f32,
    pub qx: f32,
    pub qy: f32,
    pub qz: f32,
}
impl Viewpoint {
    pub fn from(values: Vec<f32>) -> Self {
        assert_eq!(values.len(), 7, "Viewpoint must have 7 values, provided {} instead", values.len());

        Self {
            tx: values[0],
            ty: values[1],
            tz: values[2],
            qw: values[3],
            qx: values[4],
            qy: values[5],
            qz: values[6],
        }
    }
}
impl Default for Viewpoint {
    fn default() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            tz: 0.0,
            qw: 1.0,
            qx: 0.0,
            qy: 0.0,
            qz: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Ascii,
    Binary,
    BinaryCompressed,
    BinarySCompressed,
}
impl Encoding {
    pub fn as_str(&self) -> &str {
        match self {
            Encoding::Ascii => "ascii",
            Encoding::Binary => "binary",
            Encoding::BinaryCompressed => "binary_compressed",
            Encoding::BinarySCompressed => "binaryscompressed",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "ascii" => Some(Encoding::Ascii),
            "binary" => Some(Encoding::Binary),
            "binary_compressed" => Some(Encoding::BinaryCompressed),
            "binaryscompressed" => Some(Encoding::BinarySCompressed),
            _ => None,
        }
    }
}
impl Default for Encoding {
    fn default() -> Self {
        Encoding::BinaryCompressed
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldMeta {
    pub name: String,
    pub dtype: Dtype,
    pub count: usize,
}
impl FieldMeta {
    fn get_size(&self) -> usize {
        self.dtype.get_size()
    }

    fn get_type(&self) -> &str {
        self.dtype.get_type()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSchema(pub Vec<FieldMeta>);
impl FieldSchema {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, FieldMeta> {
        self.0.iter()
    }
}

impl Index<usize> for FieldSchema {
    type Output = FieldMeta;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for FieldSchema {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IntoIterator for FieldSchema {
    type Item = FieldMeta;
    type IntoIter = std::vec::IntoIter<FieldMeta>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a FieldSchema {
    type Item = &'a FieldMeta;
    type IntoIter = std::slice::Iter<'a, FieldMeta>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut FieldSchema {
    type Item = &'a mut FieldMeta;
    type IntoIter = std::slice::IterMut<'a, FieldMeta>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl FromIterator<(String, Dtype, usize)> for FieldSchema {
    fn from_iter<I: IntoIterator<Item = (String, Dtype, usize)>>(iter: I) -> Self {
        let schema = iter
            .into_iter()
            .map(|(name, dtype, count)| FieldMeta { name, dtype, count })
            .collect();
        FieldSchema(schema)
    }
}

impl FromIterator<FieldMeta> for FieldSchema {
    fn from_iter<I: IntoIterator<Item = FieldMeta>>(iter: I) -> Self {
        let schema = iter.into_iter().collect();
        FieldSchema(schema)
    }
}

impl<'a> FromIterator<(&'a str, Dtype, usize)> for FieldSchema {
    fn from_iter<I: IntoIterator<Item = (&'a str, Dtype, usize)>>(iter: I) -> Self {
        iter.into_iter()
            .map(|(name, dtype, count)| (name.to_string(), dtype, count))
            .collect()
    }
}

impl<'a> FromIterator<&'a FieldMeta> for FieldSchema {
    fn from_iter<I: IntoIterator<Item = &'a FieldMeta>>(iter: I) -> Self {
        Self {
            0: iter.into_iter()
                .map(|fm| fm.to_owned())
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_get_size() {
        assert_eq!(Dtype::U8.get_size(), 1);
        assert_eq!(Dtype::U16.get_size(), 2);
        assert_eq!(Dtype::U32.get_size(), 4);
        assert_eq!(Dtype::U64.get_size(), 8);
        assert_eq!(Dtype::I8.get_size(), 1);
        assert_eq!(Dtype::I16.get_size(), 2);
        assert_eq!(Dtype::I32.get_size(), 4);
        assert_eq!(Dtype::I64.get_size(), 8);
        assert_eq!(Dtype::F32.get_size(), 4);
        assert_eq!(Dtype::F64.get_size(), 8);
    }

    #[test]
    fn test_dtype_get_type() {
        assert_eq!(Dtype::U8.get_type(), "U");
        assert_eq!(Dtype::U16.get_type(), "U");
        assert_eq!(Dtype::U32.get_type(), "U");
        assert_eq!(Dtype::U64.get_type(), "U");
        assert_eq!(Dtype::I8.get_type(), "I");
        assert_eq!(Dtype::I16.get_type(), "I");
        assert_eq!(Dtype::I32.get_type(), "I");
        assert_eq!(Dtype::I64.get_type(), "I");
        assert_eq!(Dtype::F32.get_type(), "F");
        assert_eq!(Dtype::F64.get_type(), "F");
    }

    #[test]
    fn test_encoding_as_str() {
        assert_eq!(Encoding::Ascii.as_str(), "ascii");
        assert_eq!(Encoding::Binary.as_str(), "binary");
        assert_eq!(Encoding::BinaryCompressed.as_str(), "binary_compressed");
        assert_eq!(Encoding::BinarySCompressed.as_str(), "binaryscompressed");
    }

    #[test]
    fn test_encoding_from_str() {
        assert_eq!(Encoding::from_str("ascii"), Some(Encoding::Ascii));
        assert_eq!(Encoding::from_str("binary"), Some(Encoding::Binary));
        assert_eq!(Encoding::from_str("binary_compressed"), Some(Encoding::BinaryCompressed));
        assert_eq!(Encoding::from_str("binaryscompressed"), Some(Encoding::BinarySCompressed));
        assert_eq!(Encoding::from_str("foobar"), None);
    }

    #[test]
    fn test_metadata_new() {
        let meta = Metadata::new();
        assert_eq!(meta.fields, FieldSchema::new());
        assert_eq!(meta.width, 0);
        assert_eq!(meta.height, 1);
        assert_eq!(meta.viewpoint, Viewpoint::default());
        assert_eq!(meta.npoints, 0);
        assert_eq!(meta.encoding, Encoding::default());
    }
}
use std::{iter::FromIterator, ops::{Index, IndexMut}};
use std::sync::{Arc, RwLock};

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

pub type SharedMetadata = Arc<RwLock<Metadata>>;

impl Metadata {
    /// Constructs a new `Metadata` from the provided parameters.
    ///
    /// # Parameters
    /// - `names`: The names of the fields.
    /// - `types`: The types of the fields as strings (e.g. "U", "I", "F").
    /// - `sizes`: The size (in bytes) of each field type.
    /// - `counts`: Optional field counts (defaulting to 1 for each field if not provided).
    /// - `width`, `height`: Dimensions for organizing the point cloud.
    /// - `npoints`: Total number of points.
    /// - `viewpoint`: Optional viewpoint data.
    /// - `encoding`: Optional encoding type (e.g. "binary_compressed").
    /// - `version`: Optional version string (defaults to "0.7").
    pub fn new(
        names: Vec<String>,
        types: Vec<String>,
        sizes: Vec<usize>,
        counts: Option<Vec<usize>>,
        width: usize,
        height: usize,
        npoints: usize,
        viewpoint: Option<Vec<f32>>,
        encoding: Option<&str>,
        version: Option<&str>,
    ) -> Self {
        let fields = names.iter()
            .zip(types.iter().zip(sizes.iter()))
            .zip(counts.unwrap_or(vec![1; names.len()]).iter())
            .map(|((name, (t, s)), c)| {
                FieldMeta {
                    name: name.to_string(),
                    dtype: Dtype::from_type_size(t, s),
                    count: *c,
                }
            })
            .collect();
        let viewpoint = viewpoint.map(|vp| Viewpoint::from(vp)).unwrap_or_default();
        let encoding = Encoding::from_str(encoding.unwrap_or("binary_compressed")).unwrap();
        Self {
            fields,
            width,
            height,
            npoints,
            viewpoint,
            encoding,
            version: version.unwrap_or("0.7").to_string(),
        }
    }

    /// Creates a new `Metadata` instance by cloning the contents of a shared metadata reference.
    pub fn from_shared(shared: SharedMetadata) -> Self {
        shared.read().unwrap().clone()
    }

    /// Trims the metadata to the specified number of points.
    pub fn trim(&mut self, n: usize) {
        self.npoints = n;

        // change width, height to maintain same aspect ratio with new npoints
        let new_width = (n as f32 / self.height as f32).ceil() as usize;
        self.width = new_width;
        self.height = (n as f32 / new_width as f32).ceil() as usize;
    }
}

impl Default for Metadata {
    fn default() -> Self {
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

/// Represents the data type of a field.
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
    /// Returns the size (in bytes) for this data type.
    pub fn get_size(&self) -> usize {
        match self {
            Dtype::U8 | Dtype::I8 => 1,
            Dtype::U16 | Dtype::I16 => 2,
            Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
            Dtype::U64 | Dtype::I64 | Dtype::F64 => 8,
        }
    }

    /// Returns the type as a string ("U" for unsigned, "I" for integer, "F" for float).
    pub fn get_type(&self) -> &str {
        match self {
            Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 => "U",
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 => "I",
            Dtype::F32 | Dtype::F64 => "F",
        }
    }

    /// Constructs a `Dtype` from a type string and size.
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

    /// Returns a corresponding `Dtype` given a NumPy dtype name.
    pub fn from_numpy_dtype(dtype: &str) -> Option<Self> {
        match dtype {
            "uint8" => Some(Dtype::U8),
            "uint16" => Some(Dtype::U16),
            "uint32" => Some(Dtype::U32),
            "uint64" => Some(Dtype::U64),
            "int8" => Some(Dtype::I8),
            "int16" => Some(Dtype::I16),
            "int32" => Some(Dtype::I32),
            "int64" => Some(Dtype::I64),
            "float32" => Some(Dtype::F32),
            "float64" => Some(Dtype::F64),
            _ => None,
        }
    }

    /// Returns the NumPy dtype string corresponding to this data type.
    pub fn as_numpy_dtype(&self) -> &'static str {
        match self {
            Dtype::U8 => "uint8",
            Dtype::U16 => "uint16",
            Dtype::U32 => "uint32",
            Dtype::U64 => "uint64",
            Dtype::I8 => "int8",
            Dtype::I16 => "int16",
            Dtype::I32 => "int32",
            Dtype::I64 => "int64",
            Dtype::F32 => "float32",
            Dtype::F64 => "float64",
        }
    }

}
impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A trait for data types that can be stored in fields.
/// The trait requires that the type be copyable and also provides a static dtype.
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

/// Represents the viewpoint for the point cloud.
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
    /// Constructs a new `Viewpoint` from a vector of 7 values.
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

    /// Returns the viewpoint as a vector of f32 values.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.tx, self.ty, self.tz, self.qw, self.qx, self.qy, self.qz]
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
impl std::fmt::Display for Viewpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tx: {:.2}, ty: {:.2}, tz: {:.2}, qw: {:.2}, qx: {:.2}, qy: {:.2}, qz: {:.2}",
            self.tx, self.ty, self.tz, self.qw, self.qx, self.qy, self.qz
        )
    }
}

/// Represents the encoding format of the point cloud data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Ascii,
    Binary,
    BinaryCompressed,
}
impl Encoding {
    /// Returns the encoding as a string.
    pub fn as_str(&self) -> &str {
        match self {
            Encoding::Ascii => "ascii",
            Encoding::Binary => "binary",
            Encoding::BinaryCompressed => "binary_compressed",
        }
    }

    /// Creates an `Encoding` from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "ascii" => Some(Encoding::Ascii),
            "binary" => Some(Encoding::Binary),
            "binary_compressed" => Some(Encoding::BinaryCompressed),
            _ => None,
        }
    }
}
impl Default for Encoding {
    fn default() -> Self {
        Encoding::BinaryCompressed
    }
}

/// Metadata about a single field in the point cloud.
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

/// A schema representing a collection of field metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSchema(pub Vec<FieldMeta>);
impl FieldSchema {
    /// Creates an empty `FieldSchema`.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Returns true if the schema contains no fields.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the number of fields in the schema.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns an iterator over the field metadata.
    pub fn iter(&self) -> std::slice::Iter<'_, FieldMeta> {
        self.0.iter()
    }
}

impl std::fmt::Display for FieldSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fields = self.0.iter()
            .map(|fm| format!(" - {}[{}] - {}", fm.name, fm.count, fm.dtype))
            .collect::<Vec<String>>()
            .join("\n");
        write!(f, "{}", fields)
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
    }

    #[test]
    fn test_encoding_from_str() {
        assert_eq!(Encoding::from_str("ascii"), Some(Encoding::Ascii));
        assert_eq!(Encoding::from_str("binary"), Some(Encoding::Binary));
        assert_eq!(Encoding::from_str("binary_compressed"), Some(Encoding::BinaryCompressed));
        assert_eq!(Encoding::from_str("foobar"), None);
    }

    #[test]
    fn test_metadata_default() {
        let meta = Metadata::default();
        assert_eq!(meta.fields, FieldSchema::new());
        assert_eq!(meta.width, 0);
        assert_eq!(meta.height, 1);
        assert_eq!(meta.viewpoint, Viewpoint::default());
        assert_eq!(meta.npoints, 0);
        assert_eq!(meta.encoding, Encoding::default());
    }
}
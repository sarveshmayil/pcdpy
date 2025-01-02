#[derive(Debug, Clone)]
pub struct Metadata {
    pub fields: Vec<FieldMeta>,
    pub width: usize,
    pub height: usize,
    pub viewpoint: Viewpoint,
    pub npoints: usize,
    pub encoding: Encoding,
}

impl Metadata {
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
            width: 0,
            height: 1,
            viewpoint: Viewpoint::default(),
            npoints: 0,
            encoding: Encoding::default(),
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
}

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
    fn as_str(&self) -> &str {
        match self {
            Encoding::Ascii => "ascii",
            Encoding::Binary => "binary",
            Encoding::BinaryCompressed => "binary_compressed",
            Encoding::BinarySCompressed => "binaryscompressed",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
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
        assert_eq!(meta.fields, Vec::<FieldMeta>::new());
        assert_eq!(meta.width, 0);
        assert_eq!(meta.height, 1);
        assert_eq!(meta.viewpoint, Viewpoint::default());
        assert_eq!(meta.npoints, 0);
        assert_eq!(meta.encoding, Encoding::default());
    }
}
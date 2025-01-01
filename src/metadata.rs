#[derive(Debug, Clone)]
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

impl PartialEq for Encoding {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

#[derive(Debug, Clone)]
pub struct Metadata {
    pub fields: Vec<String>,
    pub dsize: Vec<usize>,
    pub dtype: Vec<String>,
    pub count: Vec<usize>,
    pub width: usize,
    pub height: usize,
    pub viewpoint: [f32; 7],
    pub points: usize,
    pub encoding: Encoding,
}

impl Metadata {
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
            dsize: Vec::new(),
            dtype: Vec::new(),
            count: Vec::new(),
            width: 0,
            height: 1,
            viewpoint: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            points: 0,
            encoding: Encoding::BinaryCompressed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(meta.fields, Vec::<String>::new());
        assert_eq!(meta.dsize, Vec::<usize>::new());
        assert_eq!(meta.dtype, Vec::<String>::new());
        assert_eq!(meta.count, Vec::<usize>::new());
        assert_eq!(meta.width, 0);
        assert_eq!(meta.height, 1);
        assert_eq!(meta.viewpoint, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(meta.points, 0);
        assert_eq!(meta.encoding, Encoding::BinaryCompressed);
    }
}
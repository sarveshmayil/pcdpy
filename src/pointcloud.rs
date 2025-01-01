use std::collections::HashMap;
use anyhow::Result;
use crate::fielddata::FieldData;


#[derive(Debug, Clone)]
enum Encoding {
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

#[derive(Debug, Clone)]
pub struct Metadata {
    pub fields: Vec<String>,
    pub dsize: Vec<usize>,
    pub dtype: Vec<char>,
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


#[derive(Debug, Clone)]
pub struct PointCloud {
    pub fields: HashMap<String, FieldData>,
    pub metadata: Metadata,
}

impl PointCloud {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            metadata: Metadata::new(),
        }
    }

    /// Return number of points in PointCloud
    pub fn len(&self) -> usize {
        self.metadata.points
    }

    /// Read data from PCD file and return a new PointCloud
    pub fn from_pcd_file(path: &str) -> Result<Self> {
        todo!()
    }

    /// Write PointCloud data to a PCD file
    pub fn to_pcd_file(&self, path: &str) -> Result<()> {
        todo!()
    }
}
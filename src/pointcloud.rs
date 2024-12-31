use std::collections::HashMap;

use anyhow::Result;

use crates::fielddata::FieldData

mod fielddata;

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
            "ascii" => Some(Encoding:Ascii),
            "binary" => Some(Encoding::Binary),
            "binary_compressed" => Some(Encoding::BinaryCompressed),
            "binaryscompressed" => Some(Encoding::BinarySCompressed),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Metadata {
    pub fields: [String],
    pub dsize: [usize],
    pub dtype: [char],
    pub count: [usize],
    pub width: usize,
    pub height: usize,
    pub viewpoint: [f32; 7],
    pub points: usize,
    pub encoding: Encoding,
}

impl Metadata {
    pub fn new() -> Self {
        Self {
            fields: [],
            dsize: [],
            dtype: [],
            count: 0,
            width: 0,
            height: 1,
        }
    }
}


#[derive(Debug, Clone)]
pub struct PointCloud {
    pub fields: HashMap<String, FieldData>,
    pub points: usize,
    pub width: usize,
    pub height: usize,
    version: String,
    viewpoint: [f32; 7],
    encoding: Encoding,
}

impl PointCloud {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            points: 0,
            width: 0,
            height: 1,
            version: String::from("0.7"),
            viewpoint: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            encoding: Encoding::BinaryCompressed,
        }
    }

    /// Return number of points in PointCloud
    pub fn len(&self) -> usize {
        self.points
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
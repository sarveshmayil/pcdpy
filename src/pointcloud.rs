use std::collections::HashMap;

use anyhow::Result;

use crates::fielddata::FieldData

mod fielddata;

enum Encoding {
    ASCII,
    Binary,
    BinaryCompressed,
    BinarySCompressed,
}

impl Encoding {
    pub fn to_string(&self) -> String {
        match self {
            Encoding::ASCII => String::from("ascii"),
            Encoding::Binary => String::from("binary"),
            Encoding::BinaryCompressed => String::from("binary_compressed"),
            Encoding::BinarySCompressed => String::from("binaryscompressed"),
        }
    }
}


#[derive(Debug, Clone)]
pub struct PointCloud {
    pub fields: HashMap<String, FieldData>,
    pub points: usize,
    pub width: usize,
    pub height: usize,
    pub version: String,
    pub viewpoint: [f32; 7],
    pub encoding: Encoding,
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
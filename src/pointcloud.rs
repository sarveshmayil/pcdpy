use std::collections::HashMap;
use anyhow::Result;
use crate::fielddata::FieldData;
use crate::metadata::Metadata;


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
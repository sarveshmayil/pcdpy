use std::collections::HashMap;

use anyhow::Result;

mod fielddata;

#[derive(Debug, Clone)]
pub struct PointCloud {
    pub fields: HashMap<String, FieldData>,
}

impl PointCloud {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    /// Return number of points in PointCloud
    /// For now, we just return the number of points in the first field
    pub fn len(&self) -> usize {
        if let Some((_, field1)) = self.fields.iter().next() {
            field1.len()
        } else {
            0
        }
    }

    pub fn from_pcd_file(path: &str) -> Result<Self> {
        todo!()
    }

    pub fn to_pcd_file(&self, path: &str) -> Result<()> {
        todo!()
    }
}
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use anyhow::Result;
use crate::fielddata::FieldData;
use crate::metadata::Metadata;
use crate::utils::load_metadata;


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

    /// Check if PointCloud metadata matches field data
    pub fn check_metadata(&self) -> Result<bool> {
        if self.metadata.fields.len() != self.fields.len() {
            anyhow::bail!("Metadata field count does not match field count");
        }

        for (field_name, field_data) in &self.fields {
            if let Some(field_meta) = self.metadata.fields.iter().find(|f| f.name == *field_name) {
                if field_data.dtype() != field_meta.dtype {
                    anyhow::bail!("Field '{}' dtype mismatch: expected {:?}, got {:?}", 
                        field_name, field_meta.dtype, field_data.dtype());
                }
                if field_data.len() != self.metadata.npoints {
                    anyhow::bail!("Field '{}' length mismatch: expected {}, got {}", 
                        field_name, self.metadata.npoints, field_data.len());
                }
            } else {
                anyhow::bail!("Field '{}' exists in data but not in metadata", field_name);
            }
        }
        Ok(true)
    }

    /// Return number of points in PointCloud
    pub fn len(&self) -> usize {
        self.metadata.npoints
    }

    /// Read data from PCD file and return a new PointCloud
    pub fn from_pcd_file(path: &str) -> Result<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let metadata = load_metadata(&mut file)?;
        let mut pc = PointCloud::new();
        pc.metadata = metadata;
        Ok(pc)
    }

    /// Write PointCloud data to a PCD file
    pub fn to_pcd_file(&self, path: &str) -> Result<()> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pointcloud() {
        let pc = PointCloud::new();
        assert_eq!(pc.len(), 0);
        assert!(pc.fields.is_empty());
    }
}
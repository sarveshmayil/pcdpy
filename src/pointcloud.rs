use std::{collections::HashMap, fs::File, io::{BufReader, BufWriter}};
use ndarray::Array1;
use anyhow::Result;
use crate::fielddata::FieldData;
use crate::metadata::{Dtype, Metadata, Encoding, SharedMetadata};
use crate::utils::load_metadata;
use crate::io;


#[derive(Debug, Clone)]
pub struct PointCloud {
    pub fields: HashMap<String, FieldData>,
    pub metadata: SharedMetadata,
}

impl PointCloud {
    /// Creates a new PointCloud from the provided metadata.
    pub fn new(md: &Metadata) -> Self {
        let npoints = md.npoints;
        let shared_md = std::sync::Arc::new(std::sync::RwLock::new(md.clone()));
        let mut fields_map = HashMap::new();
        for f in &md.fields {
            fields_map.insert(f.name.clone(), FieldData::new(f.dtype, npoints, f.count));
        }
        Self {
            fields: fields_map,
            metadata: shared_md,
        }
    }

    /// Creates an empty PointCloud with the given metadata.
    pub fn empty(md: &Metadata) -> Self {
        let shared_md = std::sync::Arc::new(std::sync::RwLock::new(md.clone()));
        Self {
            fields: HashMap::new(),
            metadata: shared_md,
        }
    }

    /// Check if PointCloud metadata matches field data
    pub fn check_pointcloud(&self) -> Result<()> {
        let md = self.metadata.read().unwrap();

        anyhow::ensure!(md.height * md.width == md.npoints, "Metadata height x width does not match npoints");
        anyhow::ensure!(md.fields.len() == self.fields.len(), "Metadata field count does not match field count");

        for (field_name, field_data) in &self.fields {
            if let Some(field_meta) = md.fields.iter().find(|f| f.name == *field_name) {
                anyhow::ensure!(field_data.len() == md.npoints, "Field '{}' length does not match npoints", field_name);
                anyhow::ensure!(field_data.dtype() == field_meta.dtype, "Field '{}' dtype does not match metadata", field_name);
            } else {
                anyhow::bail!("Field '{}' exists in data but not in metadata", field_name);
            }
        }
        Ok(())
    }

    /// Return number of points in PointCloud
    pub fn len(&self) -> usize {
        let md = self.metadata.read().unwrap();
        md.npoints
    }

    /// Read data from PCD file and return a new PointCloud
    pub fn from_pcd_file(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let md = load_metadata(&mut reader)?;
        let mut pc = PointCloud::new(&md);
        // Cache metadata locally to avoid repeated locking.
        let md_cached = md;

        match md_cached.encoding {
            Encoding::Ascii => {
                // For each point, read a non-empty line.
                for row_idx in 0..md_cached.npoints {
                    let line = io::read_nonempty_line(&mut reader)?;
                    let values: Vec<&str> = line.split_ascii_whitespace().collect();
                    let expected_num_values: usize = md_cached.fields.iter().map(|f| f.count).sum();
                    if values.len() != expected_num_values {
                        anyhow::bail!("Invalid data line: expected {} values, got {}", expected_num_values, values.len());
                    }
                    let mut values_iter = values.into_iter();
                    for field_meta in md_cached.fields.iter() {
                        match field_meta.dtype {
                            Dtype::U8 => {
                                let vals: Vec<u8> = values_iter.by_ref()
                                    .take(field_meta.count)
                                    .map(|v| v.parse().unwrap())
                                    .collect();
                                let array = Array1::from(vals);
                                pc.fields.get_mut(&field_meta.name).unwrap().assign_row(row_idx, &array);
                            }
                            Dtype::U16 => {
                                let vals: Vec<u16> = values_iter.by_ref()
                                    .take(field_meta.count)
                                    .map(|v| v.parse().unwrap())
                                    .collect();
                                let array = Array1::from(vals);
                                pc.fields.get_mut(&field_meta.name).unwrap().assign_row(row_idx, &array);
                            }
                            Dtype::U32 => {
                                let vals: Vec<u32> = values_iter.by_ref()
                                    .take(field_meta.count)
                                    .map(|v| v.parse().unwrap())
                                    .collect();
                                let array = Array1::from(vals);
                                pc.fields.get_mut(&field_meta.name).unwrap().assign_row(row_idx, &array);
                            }
                            Dtype::U64 => {
                                let vals: Vec<u64> = values_iter.by_ref()
                                    .take(field_meta.count)
                                    .map(|v| v.parse().unwrap())
                                    .collect();
                                let array = Array1::from(vals);
                                pc.fields.get_mut(&field_meta.name).unwrap().assign_row(row_idx, &array);
                            }
                            Dtype::I8 => {
                                let vals: Vec<i8> = values_iter.by_ref()
                                    .take(field_meta.count)
                                    .map(|v| v.parse().unwrap())
                                    .collect();
                                let array = Array1::from(vals);
                                pc.fields.get_mut(&field_meta.name).unwrap().assign_row(row_idx, &array);
                            }
                            Dtype::I16 => {
                                let vals: Vec<i16> = values_iter.by_ref()
                                    .take(field_meta.count)
                                    .map(|v| v.parse().unwrap())
                                    .collect();
                                let array = Array1::from(vals);
                                pc.fields.get_mut(&field_meta.name).unwrap().assign_row(row_idx, &array);
                            }
                            Dtype::I32 => {
                                let vals: Vec<i32> = values_iter.by_ref()
                                    .take(field_meta.count)
                                    .map(|v| v.parse().unwrap())
                                    .collect();
                                let array = Array1::from(vals);
                                pc.fields.get_mut(&field_meta.name).unwrap().assign_row(row_idx, &array);
                            }
                            Dtype::I64 => {
                                let vals: Vec<i64> = values_iter.by_ref()
                                    .take(field_meta.count)
                                    .map(|v| v.parse().unwrap())
                                    .collect();
                                let array = Array1::from(vals);
                                pc.fields.get_mut(&field_meta.name).unwrap().assign_row(row_idx, &array);
                            }
                            Dtype::F32 => {
                                let vals: Vec<f32> = values_iter.by_ref()
                                    .take(field_meta.count)
                                    .map(|v| v.parse().unwrap())
                                    .collect();
                                let array = Array1::from(vals);
                                pc.fields.get_mut(&field_meta.name).unwrap().assign_row(row_idx, &array);
                            }
                            Dtype::F64 => {
                                let vals: Vec<f64> = values_iter.by_ref()
                                    .take(field_meta.count)
                                    .map(|v| v.parse().unwrap())
                                    .collect();
                                let array = Array1::from(vals);
                                pc.fields.get_mut(&field_meta.name).unwrap().assign_row(row_idx, &array);
                            }
                        }
                    }
                }
            }
            Encoding::Binary => {
                let total_size: usize = md_cached.fields.iter().map(|f| f.dtype.get_size() * f.count).sum();
                for row_idx in 0..md_cached.npoints {
                    let data_buffer = io::read_exact_chunk(&mut reader, total_size)?;
                    let mut offset = 0;
                    for field_meta in md_cached.fields.iter() {
                        let field_bytes = field_meta.dtype.get_size() * field_meta.count;
                        let chunk = &data_buffer[offset..offset + field_bytes];
                        offset += field_bytes;
                        pc.fields.get_mut(&field_meta.name).unwrap().assign_row_from_buffer(row_idx, chunk);
                    }
                }
            }
            Encoding::BinaryCompressed => {
                let uncompressed_buf = io::read_compressed_buffer(&mut reader)?;
                let mut offset = 0;
                for field_meta in md_cached.fields.iter() {
                    let block_size = field_meta.count * field_meta.dtype.get_size() * md_cached.npoints;
                    let slice = &uncompressed_buf[offset..offset + block_size];
                    offset += block_size;
                    pc.fields.get_mut(&field_meta.name).unwrap().assign_from_buffer(slice);
                }
            }
        }

        Ok(pc)
    }

    /// Writes the PointCloud data to a PCD file.
    pub fn to_pcd_file(&self, path: &str) -> Result<()> {
        use std::io::Write;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        {
            // Get a read lock on the metadata once.
            let md = self.metadata.read().unwrap();
            io::write_header(&mut writer, &md)?;
            match md.encoding {
                Encoding::Ascii => io::write_ascii_data(&mut writer, self)?,
                Encoding::Binary => io::write_binary_data(&mut writer, self)?,
                Encoding::BinaryCompressed => io::write_compressed_data(&mut writer, self)?,
            }
        }
        writer.flush()?;
        Ok(())
    }
}
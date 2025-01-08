use std::{collections::HashMap, fs::File, io::{Read, BufRead, BufReader}};
use ndarray::Array1;
use anyhow::Result;
use byteorder::{ReadBytesExt, LittleEndian};
use std::sync::{Arc, Mutex};
use crate::fielddata::FieldData;
use crate::metadata::{Dtype, Metadata, Encoding, SharedMetadata};
use crate::utils::load_metadata;


#[derive(Debug, Clone)]
pub struct PointCloud {
    pub fields: HashMap<String, FieldData>,
    pub metadata: SharedMetadata,
}

impl PointCloud {
    pub fn new(md: &Metadata) -> Self {
        let npoints = md.npoints;
        let shared_md = Arc::new(Mutex::new(md.clone()));
        let mut fields_map = HashMap::new();
        for f in &md.fields {
            fields_map.insert(f.name.clone(), FieldData::new(f.dtype, npoints, f.count));
        }
        Self {
            fields: fields_map,
            metadata: shared_md,
        }
    }

    pub fn empty(md: &Metadata) -> Self {
        let shared_md = Arc::new(Mutex::new(md.clone()));
        Self {
            fields: HashMap::new(),
            metadata: shared_md,
        }
    }

    /// Check if PointCloud metadata matches field data
    pub fn check_pointcloud(&self) -> Result<()> {
        let md = self.metadata.lock().unwrap();

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
        let md = self.metadata.lock().unwrap();
        md.npoints
    }

    /// Read data from PCD file and return a new PointCloud
    pub fn from_pcd_file(path: &str) -> Result<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let md = load_metadata(&mut file)?;
        let mut pc = PointCloud::new(&md);

        match md.encoding {
            Encoding::Ascii => {
                for row_idx in 0..md.npoints {
                    pc.read_line(&mut file, &mut (row_idx as usize))?;
                }
            }
            Encoding::Binary => {
                for row_idx in 0..md.npoints {
                    pc.read_chunk(&mut file, &mut (row_idx as usize))?;
                }
            }
            Encoding::BinaryCompressed => {
                pc.read_compressed(&mut file)?;
            }
        }

        Ok(pc)
    }

    /// Write PointCloud data to a PCD file
    pub fn to_pcd_file(&self, path: &str) -> Result<()> {
        todo!()
    }

    fn read_line(&mut self, bufreader: &mut BufReader<File>, idx: &mut usize) -> Result<()> {
        let mut line = String::new();
        loop {
            let bytes_read = bufreader.read_line(&mut line)?;
            if bytes_read == 0 {
                anyhow::bail!("Unexpected EOF while reading data line");
            }
            if !line.trim().is_empty() {
                break;
            }
            line.clear();
        }

        // Parse data line, throw error if invalid
        let values = line.split_ascii_whitespace().collect::<Vec<&str>>();
        let md = self.metadata.lock().unwrap();
        let expected_num_values: usize = md.fields.iter().map(|f| f.count).sum();
        if values.len() != expected_num_values {
            anyhow::bail!("Invalid data line: expected {} values, got {}", expected_num_values, values.len());
        }

        let mut values_iter = values.into_iter();

        for field_meta in md.fields.iter() {
            match field_meta.dtype {
                Dtype::U8 => {
                    let vals: Vec<u8> = values_iter.by_ref().take(field_meta.count).map(|v| v.parse().unwrap()).collect();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::U16 => {
                    let vals: Vec<u16> = values_iter.by_ref().take(field_meta.count).map(|v| v.parse().unwrap()).collect();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::U32 => {
                    let vals: Vec<u32> = values_iter.by_ref().take(field_meta.count).map(|v| v.parse().unwrap()).collect();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::U64 => {
                    let vals: Vec<u64> = values_iter.by_ref().take(field_meta.count).map(|v| v.parse().unwrap()).collect();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::I8 => {
                    let vals: Vec<i8> = values_iter.by_ref().take(field_meta.count).map(|v| v.parse().unwrap()).collect();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::I16 => {
                    let vals: Vec<i16> = values_iter.by_ref().take(field_meta.count).map(|v| v.parse().unwrap()).collect();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::I32 => {
                    let vals: Vec<i32> = values_iter.by_ref().take(field_meta.count).map(|v| v.parse().unwrap()).collect();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::I64 => {
                    let vals: Vec<i64> = values_iter.by_ref().take(field_meta.count).map(|v| v.parse().unwrap()).collect();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::F32 => {
                    let vals: Vec<f32> = values_iter.by_ref().take(field_meta.count).map(|v| v.parse().unwrap()).collect();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::F64 => {
                    let vals: Vec<f64> = values_iter.by_ref().take(field_meta.count).map(|v| v.parse().unwrap()).collect();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
            }
        }
        Ok(())
    }

    fn read_chunk(&mut self, bufreader: &mut BufReader<File>, idx: &mut usize) -> Result<()> {
        let md = self.metadata.lock().unwrap();
        let total_size = md.fields.iter().map(|f| f.dtype.get_size() * f.count).sum::<usize>();

        let mut data_buffer = vec![0u8; total_size];
        bufreader.read_exact(&mut data_buffer)?;
        
        // Distribute slices of data_buffer to different fields
        let mut offset = 0;
        for field_meta in md.fields.iter() {
            let field_bytes = field_meta.dtype.get_size() * field_meta.count;
            let chunk = &data_buffer[offset..offset + field_bytes];
            offset += field_bytes;

            self.fields
                .get_mut(&field_meta.name)
                .unwrap()
                .assign_row_from_buffer(*idx, chunk);
        }
        Ok(())
    }

    fn read_compressed(&mut self, bufreader: &mut BufReader<File>) -> Result<()> {
        use lzf::decompress;

        let compressed_size = bufreader.read_u32::<LittleEndian>()? as usize;
        let uncompressed_size = bufreader.read_u32::<LittleEndian>()? as usize;

        let mut compressed_buf = vec![0u8; compressed_size];
        bufreader.read_exact(&mut compressed_buf)?;

        let uncompressed_buf = vec![0u8; uncompressed_size];
        decompress(&compressed_buf, uncompressed_size).map_err(|e| anyhow::anyhow!(e))?;

        let mut offset = 0;
        let md = self.metadata.lock().unwrap();
        for field_meta in &md.fields {
            let block_size = field_meta.count * field_meta.dtype.get_size() * md.npoints;
            let slice = &uncompressed_buf[offset..offset + block_size];
            offset += block_size;
            self.fields
                .get_mut(&field_meta.name)
                .unwrap()
                .assign_from_buffer(slice);
        }
        Ok(())
    }
}
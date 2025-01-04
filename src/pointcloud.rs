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

    /// Check if PointCloud metadata matches field data
    pub fn check_pointcloud(&self) -> Result<bool> {
        let md = self.metadata.lock().unwrap();
        if md.height * md.width != md.npoints {
            anyhow::bail!("Metadata height x width does not match npoints");
        }

        if md.fields.len() != self.fields.len() {
            anyhow::bail!("Metadata field count does not match field count");
        }

        for (field_name, field_data) in &self.fields {
            if let Some(field_meta) = md.fields.iter().find(|f| f.name == *field_name) {
                if field_data.dtype() != field_meta.dtype {
                    anyhow::bail!("Field '{}' dtype mismatch: expected {:?}, got {:?}", 
                        field_name, field_meta.dtype, field_data.dtype());
                }
                if field_data.len() != md.npoints {
                    anyhow::bail!("Field '{}' length mismatch: expected {}, got {}", 
                        field_name, md.npoints, field_data.len());
                }
            } else {
                anyhow::bail!("Field '{}' exists in data but not in metadata", field_name);
            }
        }
        Ok(true)
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
        for field_meta in md.fields.iter() {
            let count_range = 0..field_meta.count;

            match field_meta.dtype {
                Dtype::U8 => {
                    let vals = count_range
                        .map(|_| bufreader.read_u8().unwrap())
                        .collect::<Vec<u8>>();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::U16 => {
                    let vals = count_range
                        .map(|_| bufreader.read_u16::<LittleEndian>().unwrap())
                        .collect::<Vec<u16>>();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::U32 => {
                    let vals = count_range
                        .map(|_| bufreader.read_u32::<LittleEndian>().unwrap())
                        .collect::<Vec<u32>>();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::U64 => {
                    let vals = count_range
                        .map(|_| bufreader.read_u64::<LittleEndian>().unwrap())
                        .collect::<Vec<u64>>();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::I8 => {
                    let vals = count_range
                        .map(|_| bufreader.read_i8().unwrap())
                        .collect::<Vec<i8>>();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::I16 => {
                    let vals = count_range
                        .map(|_| bufreader.read_i16::<LittleEndian>().unwrap())
                        .collect::<Vec<i16>>();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::I32 => {
                    let vals = count_range
                        .map(|_| bufreader.read_i32::<LittleEndian>().unwrap())
                        .collect::<Vec<i32>>();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::I64 => {
                    let vals = count_range
                        .map(|_| bufreader.read_i64::<LittleEndian>().unwrap())
                        .collect::<Vec<i64>>();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::F32 => {
                    let vals = count_range
                        .map(|_| bufreader.read_f32::<LittleEndian>().unwrap())
                        .collect::<Vec<f32>>();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
                Dtype::F64 => {
                    let vals = count_range
                        .map(|_| bufreader.read_f64::<LittleEndian>().unwrap())
                        .collect::<Vec<f64>>();
                    let array = Array1::from(vals);
                    self.fields.get_mut(&field_meta.name).unwrap().assign_row(*idx, &array);
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_pcd_file() {
        let pc = PointCloud::from_pcd_file("/Users/smayil/Downloads/binary.pcd").unwrap();

        println!("{:?}", pc.metadata);

        // assert_eq!(pc.len(), 397);
        
        println!("{}, {}", pc.fields.get("x").unwrap().len(), pc.fields.get("x").unwrap().count());
        println!("{}", pc.fields.get("x").unwrap());
    }
}
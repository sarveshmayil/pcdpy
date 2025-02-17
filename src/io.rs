use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use anyhow::Result;
use byteorder::{ReadBytesExt, LittleEndian, WriteBytesExt};

/// Reads a non-empty, non-comment line from the given BufReader.
/// Skips empty lines and lines starting with '#' and returns the first valid line.
pub fn read_nonempty_line(reader: &mut BufReader<File>) -> Result<String> {
    let mut line = String::new();
    loop {
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            anyhow::bail!("Unexpected EOF while reading line");
        }
        let trimmed = line.trim();
        if !trimmed.is_empty() && !trimmed.starts_with('#') {
            return Ok(trimmed.to_string());
        }
        line.clear();
    }
}

/// Reads exactly `size` bytes from the reader and returns them as a Vec<u8>.
pub fn read_exact_chunk(reader: &mut BufReader<File>, size: usize) -> Result<Vec<u8>> {
    let mut buffer = vec![0u8; size];
    reader.read_exact(&mut buffer)?;
    Ok(buffer)
}

/// Reads compressed data from the reader, decompresses it using LZF,
/// and returns the uncompressed data as a Vec<u8>.
pub fn read_compressed_buffer(reader: &mut BufReader<File>) -> Result<Vec<u8>> {
    use lzf::decompress;
    let compressed_size = reader.read_u32::<LittleEndian>()? as usize;
    let uncompressed_size = reader.read_u32::<LittleEndian>()? as usize;
    let mut compressed_buf = vec![0u8; compressed_size];
    reader.read_exact(&mut compressed_buf)?;
    let uncompressed_buf = decompress(&compressed_buf, uncompressed_size)
        .map_err(|e| anyhow::anyhow!(e))?;
    Ok(uncompressed_buf)
}

/// Writes the PCD header to the provided writer using metadata.
pub fn write_header<W: Write>(writer: &mut W, md: &crate::metadata::Metadata) -> Result<()> {
    // Build header fields
    writeln!(writer, "VERSION {}", md.version)?;
    
    // Fields, SIZE, TYPE, and COUNT are based on md.fields.
    let field_names: Vec<String> = md.fields.iter().map(|f| f.name.clone()).collect();
    let sizes: Vec<String> = md.fields.iter().map(|f| f.dtype.get_size().to_string()).collect();
    let types: Vec<String> = md.fields.iter().map(|f| f.dtype.get_type().to_string()).collect();
    let counts: Vec<String> = md.fields.iter().map(|f| f.count.to_string()).collect();
    
    writeln!(writer, "FIELDS {}", field_names.join(" "))?;
    writeln!(writer, "SIZE {}", sizes.join(" "))?;
    writeln!(writer, "TYPE {}", types.join(" "))?;
    writeln!(writer, "COUNT {}", counts.join(" "))?;
    
    writeln!(writer, "WIDTH {}", md.width)?;
    writeln!(writer, "HEIGHT {}", md.height)?;
    // Write viewpoint as 7 floats.
    writeln!(writer, "VIEWPOINT {} {} {} {} {} {} {}",
             md.viewpoint.tx, md.viewpoint.ty, md.viewpoint.tz,
             md.viewpoint.qw, md.viewpoint.qx, md.viewpoint.qy, md.viewpoint.qz)?;
    writeln!(writer, "POINTS {}", md.npoints)?;
    
    // DATA: Write the encoding string (all lowercase)
    let data_str = match md.encoding {
        crate::metadata::Encoding::Ascii => "ascii",
        crate::metadata::Encoding::Binary => "binary",
        crate::metadata::Encoding::BinaryCompressed => "binary_compressed",
    };
    writeln!(writer, "DATA {}", data_str)?;
    Ok(())
}

/// Writes the point cloud data in ASCII format.
/// For each point, writes one line with the values for each field separated by a space.
pub fn write_ascii_data<W: Write>(writer: &mut W, pc: &crate::pointcloud::PointCloud) -> Result<()> {
    let md = pc.metadata.read().unwrap();
    for row_idx in 0..md.npoints {
        let mut line = String::new();
        // Iterate fields in order as in metadata.
        for field_meta in md.fields.iter() {
            let field = pc.fields.get(&field_meta.name).unwrap();
            // For each field, match on dtype and extract the row as text.
            match field_meta.dtype {
                crate::metadata::Dtype::U8 => {
                    let row = field.get_row::<u8>(row_idx);
                    for v in row.iter() { line.push_str(&format!("{} ", v)); }
                }
                crate::metadata::Dtype::U16 => {
                    let row = field.get_row::<u16>(row_idx);
                    for v in row.iter() { line.push_str(&format!("{} ", v)); }
                }
                crate::metadata::Dtype::U32 => {
                    let row = field.get_row::<u32>(row_idx);
                    for v in row.iter() { line.push_str(&format!("{} ", v)); }
                }
                crate::metadata::Dtype::U64 => {
                    let row = field.get_row::<u64>(row_idx);
                    for v in row.iter() { line.push_str(&format!("{} ", v)); }
                }
                crate::metadata::Dtype::I8 => {
                    let row = field.get_row::<i8>(row_idx);
                    for v in row.iter() { line.push_str(&format!("{} ", v)); }
                }
                crate::metadata::Dtype::I16 => {
                    let row = field.get_row::<i16>(row_idx);
                    for v in row.iter() { line.push_str(&format!("{} ", v)); }
                }
                crate::metadata::Dtype::I32 => {
                    let row = field.get_row::<i32>(row_idx);
                    for v in row.iter() { line.push_str(&format!("{} ", v)); }
                }
                crate::metadata::Dtype::I64 => {
                    let row = field.get_row::<i64>(row_idx);
                    for v in row.iter() { line.push_str(&format!("{} ", v)); }
                }
                crate::metadata::Dtype::F32 => {
                    let row = field.get_row::<f32>(row_idx);
                    for v in row.iter() { line.push_str(&format!("{:.6} ", v)); }
                }
                crate::metadata::Dtype::F64 => {
                    let row = field.get_row::<f64>(row_idx);
                    for v in row.iter() { line.push_str(&format!("{:.6} ", v)); }
                }
            }
        }
        writeln!(writer, "{}", line.trim_end())?;
    }
    Ok(())
}

/// Writes the point cloud data in binary format.
/// For each point (row), writes a contiguous block of bytes (the sum over fields of (dtype size * count))
/// with little-endian encoding.
pub fn write_binary_data<W: Write>(writer: &mut W, pc: &crate::pointcloud::PointCloud) -> Result<()> {
    let md = pc.metadata.read().unwrap();
    // Total number of bytes per point.
    let total_size: usize = md.fields.iter().map(|f| f.dtype.get_size() * f.count).sum();
    for row_idx in 0..md.npoints {
        let mut row_buffer = vec![0u8; total_size];
        let mut offset = 0;
        // Iterate over fields in metadata order.
        for field_meta in md.fields.iter() {
            let field = pc.fields.get(&field_meta.name).unwrap();
            let field_bytes = field_meta.dtype.get_size() * field_meta.count;
            // For each field, match on dtype and write the row's bytes in little-endian order.
            match field_meta.dtype {
                crate::metadata::Dtype::U8 => {
                    let row = field.get_row::<u8>(row_idx);
                    for &val in row.iter() {
                        row_buffer[offset] = val;
                        offset += 1;
                    }
                }
                crate::metadata::Dtype::U16 => {
                    let row = field.get_row::<u16>(row_idx);
                    for &val in row.iter() {
                        row_buffer[offset..offset+2].copy_from_slice(&val.to_le_bytes());
                        offset += 2;
                    }
                }
                crate::metadata::Dtype::U32 => {
                    let row = field.get_row::<u32>(row_idx);
                    for &val in row.iter() {
                        row_buffer[offset..offset+4].copy_from_slice(&val.to_le_bytes());
                        offset += 4;
                    }
                }
                crate::metadata::Dtype::U64 => {
                    let row = field.get_row::<u64>(row_idx);
                    for &val in row.iter() {
                        row_buffer[offset..offset+8].copy_from_slice(&val.to_le_bytes());
                        offset += 8;
                    }
                }
                crate::metadata::Dtype::I8 => {
                    let row = field.get_row::<i8>(row_idx);
                    for &val in row.iter() {
                        row_buffer[offset] = val as u8;
                        offset += 1;
                    }
                }
                crate::metadata::Dtype::I16 => {
                    let row = field.get_row::<i16>(row_idx);
                    for &val in row.iter() {
                        row_buffer[offset..offset+2].copy_from_slice(&val.to_le_bytes());
                        offset += 2;
                    }
                }
                crate::metadata::Dtype::I32 => {
                    let row = field.get_row::<i32>(row_idx);
                    for &val in row.iter() {
                        row_buffer[offset..offset+4].copy_from_slice(&val.to_le_bytes());
                        offset += 4;
                    }
                }
                crate::metadata::Dtype::I64 => {
                    let row = field.get_row::<i64>(row_idx);
                    for &val in row.iter() {
                        row_buffer[offset..offset+8].copy_from_slice(&val.to_le_bytes());
                        offset += 8;
                    }
                }
                crate::metadata::Dtype::F32 => {
                    let row = field.get_row::<f32>(row_idx);
                    for &val in row.iter() {
                        row_buffer[offset..offset+4].copy_from_slice(&val.to_le_bytes());
                        offset += 4;
                    }
                }
                crate::metadata::Dtype::F64 => {
                    let row = field.get_row::<f64>(row_idx);
                    for &val in row.iter() {
                        row_buffer[offset..offset+8].copy_from_slice(&val.to_le_bytes());
                        offset += 8;
                    }
                }
            }
        }
        writer.write_all(&row_buffer)?;
    }
    Ok(())
}

/// Writes the point cloud data in binary compressed format.
/// The uncompressed data is built as for binary mode, then compressed using LZF.
/// The compressed size (u32) and uncompressed size (u32) are written as headers.
pub fn write_compressed_data<W: Write>(writer: &mut W, pc: &crate::pointcloud::PointCloud) -> Result<()> {
    // First, build the uncompressed data buffer field-by-field.
    let md = pc.metadata.read().unwrap();
    let uncompressed_size: usize = md.fields.iter()
        .map(|f| f.dtype.get_size() * f.count * md.npoints)
        .sum();
    let mut uncompressed_buf = Vec::with_capacity(uncompressed_size);
    // For each field (in metadata order), append its entire data as contiguous bytes.
    for field_meta in md.fields.iter() {
        let field = pc.fields.get(&field_meta.name).unwrap();
        let field_size = field_meta.dtype.get_size();
        // Iterate over all points.
        for row_idx in 0..md.npoints {
            match field_meta.dtype {
                crate::metadata::Dtype::U8 => {
                    let row = field.get_row::<u8>(row_idx);
                    uncompressed_buf.extend_from_slice(row.as_slice().unwrap());
                }
                crate::metadata::Dtype::U16 => {
                    let row = field.get_row::<u16>(row_idx);
                    for &val in row.iter() {
                        uncompressed_buf.extend_from_slice(&val.to_le_bytes());
                    }
                }
                crate::metadata::Dtype::U32 => {
                    let row = field.get_row::<u32>(row_idx);
                    for &val in row.iter() {
                        uncompressed_buf.extend_from_slice(&val.to_le_bytes());
                    }
                }
                crate::metadata::Dtype::U64 => {
                    let row = field.get_row::<u64>(row_idx);
                    for &val in row.iter() {
                        uncompressed_buf.extend_from_slice(&val.to_le_bytes());
                    }
                }
                crate::metadata::Dtype::I8 => {
                    let row = field.get_row::<i8>(row_idx);
                    // Convert i8 to u8 for writing.
                    uncompressed_buf.extend(row.iter().map(|&v| v as u8));
                }
                crate::metadata::Dtype::I16 => {
                    let row = field.get_row::<i16>(row_idx);
                    for &val in row.iter() {
                        uncompressed_buf.extend_from_slice(&val.to_le_bytes());
                    }
                }
                crate::metadata::Dtype::I32 => {
                    let row = field.get_row::<i32>(row_idx);
                    for &val in row.iter() {
                        uncompressed_buf.extend_from_slice(&val.to_le_bytes());
                    }
                }
                crate::metadata::Dtype::I64 => {
                    let row = field.get_row::<i64>(row_idx);
                    for &val in row.iter() {
                        uncompressed_buf.extend_from_slice(&val.to_le_bytes());
                    }
                }
                crate::metadata::Dtype::F32 => {
                    let row = field.get_row::<f32>(row_idx);
                    for &val in row.iter() {
                        uncompressed_buf.extend_from_slice(&val.to_le_bytes());
                    }
                }
                crate::metadata::Dtype::F64 => {
                    let row = field.get_row::<f64>(row_idx);
                    for &val in row.iter() {
                        uncompressed_buf.extend_from_slice(&val.to_le_bytes());
                    }
                }
            }
        }
    }
    // Compress the uncompressed buffer using LZF.
    let compressed_buf = lzf::compress(&uncompressed_buf)
        .map_err(|_| anyhow::anyhow!("Compression failed"))?;
    // Write compressed size and uncompressed size as u32 little-endian.
    writer.write_u32::<LittleEndian>(compressed_buf.len() as u32)?;
    writer.write_u32::<LittleEndian>(uncompressed_buf.len() as u32)?;
    // Write compressed data.
    writer.write_all(&compressed_buf)?;
    Ok(())
}

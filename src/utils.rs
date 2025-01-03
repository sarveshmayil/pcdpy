use crate::metadata::{Metadata, Encoding, Dtype, Viewpoint, FieldSchema, FieldMeta};
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use anyhow::Result;

pub fn load_metadata(bufreader: &mut BufReader<File>) -> Result<Metadata> {
    // Initialize metadata fields as None to check if they are all present in the file
    let mut version: Option<String> = None;
    let mut fields: Option<Vec<String>> = None;
    let mut sizes: Option<Vec<usize>> = None;
    let mut types: Option<Vec<String>> = None;
    let mut counts: Option<Vec<usize>> = None;
    let mut width: Option<usize> = None;
    let mut height: Option<usize> = None;
    let mut viewpoint: Option<Viewpoint> = None;
    let mut npoints: Option<usize> = None;
    let mut encoding: Option<Encoding> = None;

    loop {
        let mut line = String::new();
        let line_size = bufreader.read_line(&mut line)?;

        // Check for EOF
        if line_size == 0 {
            anyhow::bail!("Unexpected EOF while reading metadata");
        }

        // Skip comments and empty lines
        let line = match line.trim().split('#').next() {
            Some("") | None => continue,
            Some(s) => s,
        };

        // Parse metadata line, throw error if invalid
        let values = line.split_ascii_whitespace().collect::<Vec<&str>>();
        if values.is_empty() {
            anyhow::bail!("Empty line in metadata");
        }

        // Fill in metadata fields
        match values[0] {
            "VERSION" => {
                if values.len() != 2 {
                    anyhow::bail!("Invalid VERSION line: {}", line);
                }
                version = Some(values[1].to_string());
            }
            "FIELDS" => {
                if values.len() < 2 {
                    anyhow::bail!("Invalid FIELDS line: {}", line);
                }
                fields = Some(values[1..].iter().map(|s| s.to_string()).collect());
            }
            "SIZE" => {
                if values.len() < 2 {
                    anyhow::bail!("Invalid SIZE line: {}", line);
                }
                sizes = Some(values[1..].iter().map(|s| s.parse().unwrap()).collect());
            }
            "TYPE" => {
                if values.len() < 2 {
                    anyhow::bail!("Invalid TYPE line: {}", line);
                }
                types = Some(values[1..].iter().map(|s| s.to_string()).collect());
            }
            "COUNT" => {
                if values.len() < 2 {
                    anyhow::bail!("Invalid COUNT line: {}", line);
                }
                counts = Some(values[1..].iter().map(|s| s.parse().unwrap()).collect());
            }
            "WIDTH" => {
                if values.len() != 2 {
                    anyhow::bail!("Invalid WIDTH line: {}", line);
                }
                width = Some(values[1].parse().unwrap());
            }
            "HEIGHT" => {
                if values.len() != 2 {
                    anyhow::bail!("Invalid HEIGHT line: {}", line);
                }
                height = Some(values[1].parse().unwrap());
            }
            "VIEWPOINT" => {
                let vp = values[1..].iter().map(|s| s.parse().unwrap()).collect();
                viewpoint = Some(Viewpoint::from(vp));
            }
            "POINTS" => {
                if values.len() != 2 {
                    anyhow::bail!("Invalid POINTS line: {}", line);
                }
                npoints = Some(values[1].parse().unwrap());
            }
            "DATA" => {
                if values.len() != 2 {
                    anyhow::bail!("Invalid DATA line: {}", line);
                }
                encoding = Some(
                    Encoding::from_str(values[1])
                        .ok_or_else(|| anyhow::anyhow!("Invalid encoding: {}", values[1]))?
                );
                break;
            }
            _ => {
                anyhow::bail!("Invalid metadata line: {}", line);
            }
        }
    }

    // Ensure all metadata is present
    let version = version.ok_or_else(|| anyhow::anyhow!("Missing VERSION"))?;
    let fields = fields.ok_or_else(|| anyhow::anyhow!("Missing FIELDS"))?;
    let sizes = sizes.ok_or_else(|| anyhow::anyhow!("Missing SIZE"))?;
    let types = types.ok_or_else(|| anyhow::anyhow!("Missing TYPE"))?;
    let counts = counts.ok_or_else(|| anyhow::anyhow!("Missing COUNT"))?;
    let width = width.ok_or_else(|| anyhow::anyhow!("Missing WIDTH"))?;
    let height = height.ok_or_else(|| anyhow::anyhow!("Missing HEIGHT"))?;
    let viewpoint = viewpoint.unwrap_or_default();
    let npoints = npoints.ok_or_else(|| anyhow::anyhow!("Missing POINTS"))?;
    let encoding = encoding.ok_or_else(|| anyhow::anyhow!("Missing DATA encoding"))?;

    // Create field schema by zipping fields, sizes, types, and counts
    let field_schema: Result<FieldSchema> = {
        fields.iter()
            .zip(sizes.iter())
            .zip(types.iter())
            .zip(counts.iter())
            .map(|(((name, size), dtype), &count)| {
                let dtype = Dtype::from_type_size(dtype, size);
                let field_meta = FieldMeta {
                    name: name.clone(),
                    dtype,
                    count,
                };
                Ok(field_meta)
            })
            .collect()
    };

    // Construct metadata struct
    let metadata = Metadata {
        version,
        fields: field_schema?,
        width,
        height,
        viewpoint,
        npoints,
        encoding,
    };

    Ok(metadata)
}
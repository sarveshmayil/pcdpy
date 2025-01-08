use pyo3::{exceptions::{PyKeyError, PyValueError}, prelude::*, types::PySlice, IntoPyObjectExt};
use numpy::{PyArray2, PyArrayMethods};
use ndarray::s;
use crate::{fielddata::{FieldData, IntoPyObjectShaped}, pointcloud::PointCloud};
use crate::pymetadata::PyMetadata;
use crate::metadata::{FieldMeta, Dtype};

#[pyclass(name = "PointCloud")]
pub struct PyPointCloud {
    pub pc: PointCloud,
}

#[pymethods]
impl PyPointCloud {
    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<Self> {
        let pc = PointCloud::from_pcd_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyPointCloud { pc })
    }

    #[staticmethod]
    pub fn from_metadata(metadata: &Bound<'_, PyMetadata>) -> PyResult<Self> {
        let meta = metadata.borrow().inner.lock().unwrap().clone();
        let pc = PointCloud::new(&meta);
        Ok(PyPointCloud { pc })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        self.pc.to_pcd_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    fn __len__(&self) -> usize {
        self.pc.len()
    }

    fn __repr__(&self) -> String {
        let md = self.pc.metadata.lock().unwrap();
        format!("PointCloud\n Fields:\n{}\n Points: {}, Width: {}, Height: {}\n Viewpoint: {}\n Encoding: {}\n Version: {}",
            md.fields,
            md.npoints,
            md.width,
            md.height,
            md.viewpoint,
            md.encoding.as_str(),
            md.version,
        )
    }

    #[getter]
    pub fn metadata(&self) -> PyMetadata {
        PyMetadata {
            inner: self.pc.metadata.clone()
        }
    }

    /// Get a field by name
    /// Returns None if field does not exist
    /// Returns a 2D Numpy array if field exists (npoints, count)
    fn get_field<'py>(&self, py: Python<'py>, field_name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let Some(field_data) = self.pc.fields.get(field_name) {
            Ok(Some(field_data.into_pyobject(py)?))
        } else {
            Ok(None)
        }
    }
    
    /// Get a field by name, reshaped to (height, width)
    /// Returns None if field does not exist
    /// Returns a 3D Numpy array if field exists (height, width, count)
    fn get_field_shaped<'py>(&self, py: Python<'py>, field_name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        let width = self.pc.metadata.lock().unwrap().width;
        let height = self.pc.metadata.lock().unwrap().height;
        if let Some(field_data) = self.pc.fields.get(field_name) {
            Ok(Some(field_data.into_pyobject_shaped(py, width, height)?))
        } else {
            Ok(None)
        }
    }

    /// Implement __getitem__ in Python:
    ///   - If key is a str or list/tuple of str => treat as field(s).
    ///   - If key is a slice => return a *new* sliced PointCloud.
    fn __getitem__<'py>(&self, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = key.py();
        // Check if key is a slice object.
        if let Ok(slice) = key.downcast::<PySlice>() {
            let indices = slice.indices(self.pc.len() as isize)?;
            let start = indices.start as usize;
            let stop = indices.stop as usize;
            let step = indices.step as usize;

            // Create a new, sliced PointCloud
            let mut md = self.pc.metadata.lock().unwrap().clone();
            md.trim((stop - start) / step);
            let mut new_pc = PointCloud::empty(&md);

            for (field_name, field_data) in &self.pc.fields {
                let data_slice = field_data.slice(start, stop, step);
                new_pc.fields.insert(field_name.clone(), data_slice);
            }

            return Ok(PyPointCloud { pc: new_pc }.into_bound_py_any(py)?);
        }

        // Check if key is a string => return one field as a Numpy array
        else if let Ok(field_name) = key.extract::<String>() {
            if let Some(field_data) = self.pc.fields.get(&field_name) {
                // Return the field as a NumPy array
                let arr = field_data.into_pyobject(py)?;
                return Ok(arr.into_bound_py_any(py)?);
            } else { // Python KeyError
                return Err(
                    PyKeyError::new_err(format!("No field named '{}'", field_name)
                ));
            }
        }

        // Check if key is a list/tuple of strings => return 2D Numpy array
        else if let Ok(field_names) = key.extract::<Vec<String>>() {
            let ncols = self.pc.metadata.lock().unwrap().fields
                .iter()
                .filter_map(|f| {
                    if field_names.contains(&f.name) {
                        Some(f.count)
                    } else {
                        None
                    }
                })
                .sum::<usize>();
            let arr2d = PyArray2::<f64>::zeros(py, [self.pc.len(), ncols], false);
            let mut arr2d_view = unsafe { arr2d.as_array_mut() };

            let mut col_idx: usize = 0;
            for field_name in field_names {
                if let Some(field_data) = self.pc.fields.get(&field_name) {
                    let ncols = field_data.count();
                    let source_array = field_data.into_pyarray(py)?;
                    let source_view = unsafe { source_array.as_array() };
            
                    // Copy values using ndarray slice assignment
                    arr2d_view.slice_mut(s![.., col_idx..col_idx+ncols])
                        .assign(&source_view);
                    
                    col_idx += ncols;
                } else { // Python KeyError
                    return Err(
                        PyKeyError::new_err(format!("No field named '{}'", field_name)
                    ));
                }
            }
            return Ok(arr2d.into_bound_py_any(py)?);
        }

        else {
            return Err(PyKeyError::new_err("Invalid key type. Must be a str, list/tuple of str, or slice."));
        }
    }

    /// Implement __setitem__:
    ///   - If key is a str => set a field with dtype inference
    ///   - If key is a slice => partial assignment to existing fields? (optional, more complex)
    fn __setitem__<'py>(&mut self, key: &Bound<'py, PyAny>, value: &Bound<'py, PyAny>) -> PyResult<()> {
        // let py = key.py();
        if let Ok(field_name) = key.extract::<String>() {
            // Infer dtype from Numpy array and store it in PointCloud fields
            infer_and_store_field(&mut self.pc, &field_name, value)?;
            Ok(())
        } else {
            return Err(PyKeyError::new_err("Invalid key type. Must be a str, list/tuple of str, or slice."));
        }
    }
}

/// Helper functions ///

/// Infer dtype from Numpy array and store it in PointCloud fields
fn infer_and_store_field<'py>(pc: &mut PointCloud, field_name: &str, pyarray:&Bound<'py, PyAny>) -> PyResult<()> {
    if field_name.is_empty() {
        return Err(PyValueError::new_err("Field name cannot be empty"));
    }

    let (npoints, existing_field_meta) = {
        let md = pc.metadata.lock().unwrap();
        let field_meta = md.fields.iter().find(|f| f.name == field_name).cloned();
        (md.npoints, field_meta)
    };
    
    let dtype_obj = pyarray.getattr("dtype")?;
    let dtype_name: String = dtype_obj.getattr("name")?.extract()?;
    let shape = pyarray.getattr("shape")?.extract::<(usize, usize)>()?;

    if shape.0 != npoints {
        return Err(PyValueError::new_err(format!(
            "Array length mismatch: expected {}, got {}", 
            npoints, shape.0
        )));
    }

    let dtype = Dtype::from_numpy_dtype(&dtype_name)
        .ok_or_else(|| PyValueError::new_err(format!("Unsupported dtype: {}", dtype_name)))?;

    // Validate against existing field if present
    if let Some(field_meta) = existing_field_meta {
        if field_meta.dtype != dtype {
            return Err(PyValueError::new_err(format!(
                "Dtype mismatch: field has {}, array has {}", 
                field_meta.dtype.as_numpy_dtype(), dtype_name
            )));
        }
        if field_meta.count != shape.1 {
            return Err(PyValueError::new_err(format!(
                "Count mismatch: field has {}, array has {}", 
                field_meta.count, shape.1
            )));
        }
    } else {
        let mut md = pc.metadata.lock().unwrap();
        md.fields.0.push(FieldMeta {
            name: field_name.to_string(),
            dtype,
            count: shape.1,
        });
    }

    // Convert array to FieldData
    let field_data = FieldData::from_pyarray(pyarray, dtype)?;

    pc.fields.insert(field_name.to_string(), field_data);

    Ok(())
}
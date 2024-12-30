use pyo3::prelude::*;
use numpy::{PyReadOnlyArray1, PyArray2}

use crate::pointcloud::PointCloud;

mod pointcloud;

#[pyclass(name = "PointCloud")]
pub struct PyPointCloud {
    pub pc: PointCloud,
}

#[pymethods]
impl PyPointCloud {
    #[new]
    fn new() -> Self {
        PyPointCloud {
            pc: PointCloud::new(),
        }
    }

    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<Self> {
        let pc = PointCloud::from_pcd_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyPointCloud { pc })
    }

    pub fn to_file(&self, path: &str) -> PyResult<()> {
        self.pc.to_pcd_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Python-facing method to set a field, inferring dtype from the given array.
    #[pyo3(text_signature = "(self, field_name, arr)")]
    pub fn set_field(&mut self, field_name: &str, arr: &PyAny) -> PyResult<()> {
        infer_and_store_field(&mut self.pc, field_name, arr)?;
        Ok(())
    }

    #[pyo3(text_signature = "(self, field_name)")]
    pub fn get_field<'py>(&self, py: Python<'py>, field_name: &str) -> PyResult<Option<&'py PyAny>> {
        if let Some(field_data) = self.pc.fields.get(field_name) {
            let arr = field_data.to_pyarray(py)?;
            Ok(Some(arr))
        } else {
            Ok(None)
        }
    }

    /// Implement __getitem__ in Python:
    ///   - If key is a str or list of str => treat as field(s).
    ///   - If key is a slice => return a *new* sliced PointCloud.
    #[call]
    fn __getitem__(&self, key: &PyAny) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            // Check if key is a slice object.
            if let Ok(slice) = key.downcast::<PySlice>() {
                let (start, stop, step) = slice_indices(slice, self.pc.len())?;

                // Create a new, sliced PointCloud
                let mut new_pc = PointCloud::new();
                for (field_name, field_data) in &self.pc.fields {
                    let data_slice = field_data.slice(start, stop, step);
                    new_pc.fields.insert(field_name.clone(), data_slice)
                }

                return Ok(PyPointCloud { pc: new_pc }.into_py(py));
            }

            // Check if key is a string => return one field as a Numpy array
            if let Ok(field_name) = key.extract::<String>() {
                if let Some(field_data) = self.pc.fields.get(&field_name) {
                    // Return the field as a NumPy array
                    let arr = field_data.to_pyarray(py)?;
                    return Ok(arr.into_py(py));
                } else { // Python KeyError
                    return Err(
                        PyValueError::new_err(format!("No field named '{}'", field_name)
                    ));
                }
            }

            // Check if key is a list/tuple of strings => return 2D Numpy array
            if let Ok(field_names) = key.extract::<Vec<String>>() {
                // TODO: Fix array initialization
                let field_data_arr2d = PyArray2::<f32>::new()
                for field_name in field_names {
                    if let Some(field_data) = self.pc.fields.get(&field_name) {
                        todo!()
                    } else {
                        return Err(
                            PyValueError::new_err(format!("No field named '{}'", field_name)
                        ));
                    }
                }
                return Ok(field_data_arr2d.into_py(py));
            }

            Err(PyValueError::new_err("Invalid index type"))
        })
    }

    /// Implement __setitem__:
    ///   - If key is a str => set a field with dtype inference
    ///   - If key is a slice => partial assignment to existing fields? (optional, more complex)
    #[call]
    fn __setitem__(&mut self, key: &PyAny, value: &PyAny) -> PyResult<()> {
        // For now, only handles the case: pc["field_name"] = np_array
        // TODO: slice-based assignment
        if let Ok(field_name) = key.extract::<String>() {
            // infer dtype from `value` (a np array) and store it
            infer_and_store_field(&mut self.pc, &field_name, value)?;
            Ok(())
        } else {
            Err(PyValueError::new_err("Only direct field_name assignment is supported."))
        }
    }
}

/// Helper functions ///

/// Infer dtype from Numpy array and store it in PointCloud fields
fn infer_and_store_field(pc: &mut PointCloud, field_name: &str, pyarray: &PyAny) -> PyResult<()> {
    let dtype_obj = pyarray.getattr("dtype")?;
    let dtype_name: String = dtype_obj.getattr("name")?.extract()?;

    match dtype_name.as_str() {
        "uint8" => {
            let arr = pyarray.extract::<PyReadOnlyArray1<u8>>()?;
            let arr_owned = arr.as_array().to_owned();
            pc.fields.insert(field_name.to_string(), FieldData::U1(arr_owned));
        }
        "uint16" => {
            let arr = pyarray.extract::<PyReadOnlyArray1<u16>>()?;
            let arr_owned = arr.as_array().to_owned();
            pc.fields.insert(field_name.to_string(), FieldData::U2(arr_owned));
        }
        "uint32" => {
            let arr = pyarray.extract::<PyReadOnlyArray1<u32>>()?;
            let arr_owned = arr.as_array().to_owned();
            pc.fields.insert(field_name.to_string(), FieldData::U4(arr_owned));
        }
        "uint64" => {
            let arr = pyarray.extract::<PyReadOnlyArray1<u64>>()?;
            let arr_owned = arr.as_array().to_owned();
            pc.fields.insert(field_name.to_string(), FieldData::U8(arr_owned));
        }
        "int8" => {
            let arr = pyarray.extract::<PyReadOnlyArray1<i8>>()?;
            let arr_owned = arr.as_array().to_owned();
            pc.fields.insert(field_name.to_string(), FieldData::I1(arr_owned));
        }
        "int16" => {
            let arr = pyarray.extract::<PyReadOnlyArray1<i16>>()?;
            let arr_owned = arr.as_array().to_owned();
            pc.fields.insert(field_name.to_string(), FieldData::I2(arr_owned));
        }
        "int32" => {
            let arr = pyarray.extract::<PyReadOnlyArray1<i32>>()?;
            let arr_owned = arr.as_array().to_owned();
            pc.fields.insert(field_name.to_string(), FieldData::I4(arr_owned));
        }
        "int64" => {
            let arr = pyarray.extract::<PyReadOnlyArray1<i64>>()?;
            let arr_owned = arr.as_array().to_owned();
            pc.fields.insert(field_name.to_string(), FieldData::I8(arr_owned));
        }
        "float32" => {
            let arr = pyarray.extract::<PyReadOnlyArray1<f32>>()?;
            let arr_owned = arr.as_array().to_owned();
            pc.fields.insert(field_name.to_string(), FieldData::F4(arr_owned));
        }
        "float64" => {
            let arr = pyarray.extract::<PyReadOnlyArray1<f64>>()?;
            let arr_owned = arr.as_array().to_owned();
            pc.fields.insert(field_name.to_string(), FieldData::F8(arr_owned));
        }
        other => {
            return Err(PyValueError::new_err(format!("Unsupported dtype {}", other)));
        }
    }
}


/// Parse a PySlice into concrete (start, stop, step).
fn slice_indices(slice: &PySlice, length: usize) -> PyResult<(usize, usize, usize)> {
    let indices = slice.indices(length as isize)?;
    let start = indices.start as usize;
    let stop = indices.stop as usize;
    let step = indices.step as usize;
    Ok((start, stop, step))
}
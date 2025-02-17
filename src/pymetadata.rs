use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::metadata::{SharedMetadata, Encoding};

#[pyclass(name = "Metadata")]
pub struct PyMetadata {
    pub inner: SharedMetadata,
}

#[pymethods]
impl PyMetadata {
    fn __repr__(&self) -> String {
        let md = self.inner.read().unwrap();
        format!("PointCloud Metadata\n Fields:\n{}\n Points: {}, Width: {}, Height: {}\n Viewpoint: {}\n Encoding: {}\n Version: {}",
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
    fn get_fields(&self) -> Vec<String> {
        let md = self.inner.read().unwrap();
        md.fields.iter().map(|f| f.name.clone()).collect()
    }

    #[setter]
    fn set_fields(&mut self, new_fields: Vec<String>) -> PyResult<()> {
        let mut md = self.inner.write().unwrap();
        if new_fields.len() != md.fields.len() {
            return Err(PyValueError::new_err("Length mismatch with existing field schema"));
        }
        for (i, name) in new_fields.iter().enumerate() {
            md.fields[i].name = name.clone();
        }
        Ok(())
    }

    #[getter]
    fn get_width(&self) -> usize {
        self.inner.read().unwrap().width
    }

    #[getter]
    fn get_height(&self) -> usize {
        self.inner.read().unwrap().height
    }

    #[getter]
    fn get_npoints(&self) -> usize {
        self.inner.read().unwrap().npoints
    }

    #[getter]
    fn get_shape(&self) -> (usize, usize) {
        let md = self.inner.read().unwrap();
        (md.width, md.height)
    }

    #[setter]
    fn set_shape(&mut self, value: (usize, usize)) -> PyResult<()> {
        let mut md = self.inner.write().unwrap();
        if value.0 * value.1 != md.npoints {
            return Err(PyValueError::new_err("Shape must match number of points"));
        }
        md.width = value.0;
        md.height = value.1;
        Ok(())
    }

    #[getter]
    fn get_viewpoint(&self) -> (f32, f32, f32, f32, f32, f32, f32) {
        let md = self.inner.read().unwrap();
        let vp = &md.viewpoint;
        (vp.tx, vp.ty, vp.tz, vp.qw, vp.qx, vp.qy, vp.qz)
    }

    #[setter]
    fn set_viewpoint(&mut self, value: (f32, f32, f32, f32, f32, f32, f32)) {
        let mut md = self.inner.write().unwrap();
        md.viewpoint.tx = value.0;
        md.viewpoint.ty = value.1;
        md.viewpoint.tz = value.2;
        md.viewpoint.qw = value.3;
        md.viewpoint.qx = value.4;
        md.viewpoint.qy = value.5;
        md.viewpoint.qz = value.6;
    }

    #[getter]
    fn get_encoding(&self) -> String {
        self.inner.read().unwrap().encoding.as_str().to_string()
    }

    #[setter]
    fn set_encoding(&mut self, val: &str) -> PyResult<()> {
        let mut md = self.inner.write().unwrap();
        md.encoding = Encoding::from_str(val.to_lowercase().as_str())
            .ok_or_else(|| PyValueError::new_err("Invalid encoding value"))?;
        Ok(())
    }
}
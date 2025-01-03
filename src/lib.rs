use pyo3::prelude::*;

mod utils;
mod metadata;
mod fielddata;
mod pointcloud;
mod pypointcloud;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pypointcloud::PyPointCloud>()?;
    Ok(())
}

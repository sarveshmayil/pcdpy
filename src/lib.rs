use pyo3::prelude::*;

#[pyfunction]
fn factorial(n: u128) -> PyResult<u128> {
    Ok(_factorial(n))
}

fn _factorial(n: u128) -> u128 {
    if n <= 1 {
        return n
    } else {
        return n * _factorial(n - 1)
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(factorial, m)?)?;
    Ok(())
}

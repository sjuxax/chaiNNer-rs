mod clipboard;
mod convert;
mod dither;
mod pixel_art;
mod regex;
mod resize;

use image_core::{Image, NDimImage};
use image_ops::fill_alpha::{fill_alpha, FillMode};
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::convert::{IntoNumpy, LoadImage, PyImage};

/// A Python module implemented in Rust.
#[pymodule]
fn chainner_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<regex::RustRegex>()?;
    m.add_class::<regex::MatchGroup>()?;
    m.add_class::<regex::RegexMatch>()?;

    m.add_class::<clipboard::Clipboard>()?;

    m.add_class::<dither::DiffusionAlgorithm>()?;
    m.add_class::<dither::UniformQuantization>()?;
    m.add_class::<dither::PaletteQuantization>()?;
    m.add_function(wrap_pyfunction!(dither::quantize, m)?)?;
    m.add_function(wrap_pyfunction!(dither::error_diffusion_dither, m)?)?;
    m.add_function(wrap_pyfunction!(dither::ordered_dither, m)?)?;
    m.add_function(wrap_pyfunction!(dither::riemersma_dither, m)?)?;

    m.add_function(wrap_pyfunction!(pixel_art::pixel_art_upscale, m)?)?;

    m.add_class::<resize::ResizeFilter>()?;
    m.add_function(wrap_pyfunction!(resize::resize, m)?)?;

    m.add_function(wrap_pyfunction!(fill_alpha_fragment_blur, m)?)?;
    m.add_function(wrap_pyfunction!(fill_alpha_extend_color, m)?)?;
    m.add_function(wrap_pyfunction!(fill_alpha_nearest_color, m)?)?;
    m.add_function(wrap_pyfunction!(binary_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(esdf, m)?)?;
    m.add_function(wrap_pyfunction!(fast_gamma, m)?)?;

    Ok(())
}

/// Fill the transparent pixels in the given image with nearby colors.
#[pyfunction]
fn fill_alpha_fragment_blur<'py>(
    py: Python<'py>,
    img: PyImage<'py>,
    threshold: f32,
    iterations: u32,
    fragment_count: u32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let mut img = img.load_image()?;
    let result = py.detach(|| {
        fill_alpha(
            &mut img,
            threshold,
            FillMode::Fragment {
                iterations,
                fragment_count,
            },
            None,
        );
        img.into_numpy()
    });
    Ok(result.into_pyarray(py))
}

/// Fill the transparent pixels in the given image with nearby colors.
#[pyfunction]
fn fill_alpha_extend_color<'py>(
    py: Python<'py>,
    img: PyImage<'py>,
    threshold: f32,
    iterations: u32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let mut img = img.load_image()?;
    let result = py.detach(|| {
        fill_alpha(
            &mut img,
            threshold,
            FillMode::ExtendColor { iterations },
            None,
        );
        img.into_numpy()
    });
    Ok(result.into_pyarray(py))
}

/// Fill the transparent pixels in the given image with nearby colors.
#[pyfunction]
fn fill_alpha_nearest_color<'py>(
    py: Python<'py>,
    img: PyImage<'py>,
    threshold: f32,
    min_radius: u32,
    anti_aliasing: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let mut img = img.load_image()?;
    let result = py.detach(|| {
        fill_alpha(
            &mut img,
            threshold,
            FillMode::Nearest {
                min_radius,
                anti_aliasing,
            },
            None,
        );
        img.into_numpy()
    });
    Ok(result.into_pyarray(py))
}

/// Fill the transparent pixels in the given image with nearby colors.
#[pyfunction]
fn binary_threshold<'py>(
    py: Python<'py>,
    img: PyImage<'py>,
    threshold: f32,
    anti_aliasing: bool,
    extra_smoothness: Option<f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let mut img: NDimImage = img.load_image()?;
    let result = py.detach(|| {
        let aa = if anti_aliasing {
            Some(image_ops::threshold::AntiAliasing {
                extra_smoothness: extra_smoothness.unwrap_or(0.0),
            })
        } else {
            None
        };

        image_ops::threshold::binary_threshold(&mut img, threshold, aa);
        img.into_numpy()
    });
    Ok(result.into_pyarray(py))
}

/// Fill the transparent pixels in the given image with nearby colors.
#[pyfunction]
fn esdf<'py>(
    py: Python<'py>,
    img: PyImage<'py>,
    radius: f32,
    cutoff: f32,
    pre_process: bool,
    post_process: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let img: Image<f32> = img.load_image()?;
    let result = py.detach(|| {
        image_ops::esdt::esdf(&img, radius, cutoff, pre_process, post_process).into_numpy()
    });
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn fast_gamma<'py>(
    py: Python<'py>,
    img: PyImage<'py>,
    gamma: f32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let mut img = img.load_image()?;
    let result = py.detach(|| {
        image_ops::gamma::gamma_ndim(&mut img, gamma);
        img.into_numpy()
    });
    Ok(result.into_pyarray(py))
}

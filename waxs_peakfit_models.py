"""Peak fitting models for WAXS data (JAX-compatible)."""

import jax.numpy as jnp


def pseudo_voigt(x, amp, center, width, eta):
    """Pseudo-Voigt profile. Width is FWHM."""
    dx = (x - center) / width
    g = jnp.exp(-4 * jnp.log(2) * dx**2)
    l = 1.0 / (1 + 4 * dx**2)
    return amp * (eta * l + (1 - eta) * g)


def build_model(n_peaks):
    """Build n-peak pseudo-Voigt + linear background model.

    Params layout: [amp0, cen0, wid0, eta0, ..., slope, intercept]
    Total: 4*n_peaks + 2 parameters.
    """
    def model(params, x):
        y = jnp.zeros_like(x)
        for i in range(n_peaks):
            y += pseudo_voigt(x, params[4*i], params[4*i+1],
                              params[4*i+2], params[4*i+3])
        y += params[-2] * x + params[-1]
        return y
    return model


def build_residual(n_peaks):
    """Build residual function for least-squares fitting."""
    model = build_model(n_peaks)
    def residual(params, x, y):
        return model(params, x) - y
    return residual

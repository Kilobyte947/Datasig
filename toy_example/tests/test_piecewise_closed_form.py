import torch

from toy_example.toy_functions import piecewise_ramp_f, piecewise_ramp_true_L


def test_piecewise_ramp_slope_matches_true_L_inside_ramp():
    c, half_width, slope = 0.0, 2.0, 3.5
    x = torch.linspace(c - half_width, c + half_width, 101)
    y = piecewise_ramp_f(x, c, half_width, slope)
    finite_diff_slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    assert torch.allclose(finite_diff_slopes, torch.full_like(finite_diff_slopes, slope), atol=1e-6)
    assert abs(finite_diff_slopes.abs().max().item() - piecewise_ramp_true_L(slope)) < 1e-6


def test_piecewise_ramp_flat_outside_ramp():
    c, half_width, slope = 1.0, 0.5, 2.0
    x_left = torch.tensor([c - half_width - 5.0, c - half_width - 1.0])
    x_right = torch.tensor([c + half_width + 1.0, c + half_width + 5.0])
    y_left = piecewise_ramp_f(x_left, c, half_width, slope)
    y_right = piecewise_ramp_f(x_right, c, half_width, slope)
    assert torch.allclose(y_left, torch.zeros_like(y_left))
    ramp_height = slope * (2 * half_width)
    assert torch.allclose(y_right, torch.full_like(y_right, ramp_height))


def test_piecewise_ramp_true_L_is_exactly_abs_slope():
    for slope in [-5.0, -0.1, 0.0, 0.1, 5.0]:
        assert piecewise_ramp_true_L(slope) == abs(slope)


def test_piecewise_ramp_global_max_slope_matches_true_L_over_wide_domain():
    # A brute-force numeric Lipschitz estimate (max finite-difference slope
    # over a fine grid spanning well beyond the ramp) must not exceed the
    # closed-form value - the flat regions can only ever contribute slope 0.
    c, half_width, slope = -1.5, 0.8, -4.0
    x = torch.linspace(c - 5 * half_width, c + 5 * half_width, 5001)
    y = piecewise_ramp_f(x, c, half_width, slope)
    finite_diff = ((y[1:] - y[:-1]) / (x[1:] - x[:-1])).abs()
    assert finite_diff.max().item() <= piecewise_ramp_true_L(slope) + 1e-6

import torch

from signature_distance.plots import (
    plot_per_line_bar,
    plot_ratio_distribution,
    plot_reference_lines_with_metric,
)
from signature_distance.streams import make_reference_lines


def test_plot_reference_lines_with_metric_runs():
    import matplotlib
    matplotlib.use("Agg")
    image = torch.rand(28, 28)
    lines = make_reference_lines(angles_deg=(0, 90), counts=(12, 4))
    metric_values = torch.rand(lines.shape[0])
    fig = plot_reference_lines_with_metric(image, lines, metric_values, colorbar_label="fold-ratio")
    assert fig is not None


def test_plot_ratio_distribution_runs_with_tensors_and_arrays():
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    adv = torch.rand(50) + 1.0
    ctrl = torch.rand(50) * 0.2
    fig = plot_ratio_distribution(adv, ctrl)
    assert fig is not None

    fig2 = plot_ratio_distribution(np.array(adv), np.array(ctrl))
    assert fig2 is not None


def test_plot_per_line_bar_runs_and_highlights():
    import matplotlib
    matplotlib.use("Agg")
    values = {i: float(i) for i in range(5)}
    fig = plot_per_line_bar(values, highlight_key=3, ylabel="fold-ratio")
    assert fig is not None

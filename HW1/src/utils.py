import matplotlib.pyplot as plt
import numpy as np

def plot_contours_with_paths(f, xlim, ylim, paths=None, labels=None, title="Contour Plot", levels=30):
    """
    Plots 2D contour lines of the objective function and overlays optimization paths in color.

    - f: objective function (returns f_val, grad, hess)
    - xlim, ylim: (xmin, xmax), (ymin, ymax)
    - paths: list of lists of x points
    - labels: method names for legend
    """
    pass

def plot_function_values(f_paths, labels, title="Function Value per Iteration"):
    """
    Plots f(x) vs iteration number for each method with colored lines.
    """
    pass
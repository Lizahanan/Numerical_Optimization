import unittest
import sys
import os as os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.unconstrained_min import minimize
from tests.examples import (
    quad_circle, quad_ellipse, quad_rotated_ellipse,
    rosenbrock, linear, smooth_triangle
)
from src.utils import plot_contours_with_paths, plot_function_values
plot_settings = {
    "quad_circle": {
        "xlim": (-2, 2),
        "ylim": (-2, 2),
        "levels": 30  # auto-scaled
    },
    "quad_ellipse": {
        "xlim": (-2, 2),
        "ylim": (-2, 2),
        "levels": 10
    },
    "quad_rotated_ellipse": {
        "xlim": (-2, 2),
        "ylim": (-2, 2),
        "levels": 30
    },
    "rosenbrock": {
        "xlim": (-2, 2),
        "ylim": (-1, 3),
        "levels": np.logspace(0, 3.5, 30)  # good for steep sides
    },
    "linear": {
        "xlim": (-2, 2),
        "ylim": (-130, 2),
        "levels": np.linspace(-5000, 100, 30)
    },
    "smooth_triangle": {
        "xlim": (-2, 2),
        "ylim": (-2, 2),
        "levels": np.logspace(0, 4, 25)
    }
 }
class TestUnconstrainedMin(unittest.TestCase):

    def run_test_case(self, func, name, x0, max_iter, out_dir):
        gd_result = minimize(func, x0, method="GD", max_iter=max_iter,
                             )
        nt_result = minimize(func, x0, method="NT", max_iter=max_iter,
                             )

        print(f"[{name}] GD Final: x = {gd_result[0]}, f(x) = {gd_result[1]}, Success = {gd_result[2]}")
        print(f"[{name}] NT Final: x = {nt_result[0]}, f(x) = {nt_result[1]}, Success = {nt_result[2]}")

        # Save plots
        os.makedirs(out_dir, exist_ok=True)
        settings = plot_settings[name]
        xlim = settings["xlim"]
        ylim = settings["ylim"]
        levels = settings["levels"]
        plot_contours_with_paths(
            func,
            xlim=xlim, ylim=ylim,
            paths=[gd_result[3], nt_result[3]],
            labels=["GD", "NT"],
            title=f"{name} - Contour with Paths",
            levels=levels,
            save_path=os.path.join(out_dir, f"{name}_contour.png")
        )

        plot_function_values(
            [gd_result[4], nt_result[4]],
            labels=["GD", "NT"],
            title=f"{name} - Function Value vs Iteration",
            save_path=os.path.join(out_dir, f"{name}_fval.png")
        )

    def test_all(self):
        functions = {
            "quad_circle": quad_circle,
            "quad_ellipse": quad_ellipse,
            "quad_rotated_ellipse": quad_rotated_ellipse,
            "rosenbrock": rosenbrock,
            "linear": linear,
            "smooth_triangle": smooth_triangle
        }
        for name, func in functions.items():
            x0 = np.array([1.0, 1.0]) if name != 'rosenbrock' else np.array([-1.0, 2.0])
            max_iter = 100 if name != 'rosenbrock' else 10000
            out_dir = "results"
            self.run_test_case(func, name, x0, max_iter, out_dir)
           


if __name__ == "__main__":
    unittest.main()

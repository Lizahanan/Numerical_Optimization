import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
from src.unconstrained_min import minimize
from src.utils import plot_contours_with_paths, plot_function_values
from tests.examples import (
    quad_circle, quad_ellipse, quad_rotated_ellipse,
    rosenbrock, linear, smooth_triangle
)
# the plot settings for each function
plot_settings = {
    "quad_circle": {
        "xlim": (-2, 2),
        "ylim": (-2, 2),
        "levels": 30  # auto-scaled
    },
    "quad_ellipse": {
        "xlim": (-2, 2),
        "ylim": (-2, 2),
        "levels": 30
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
# Function to run the optimization and plot results
test_cases = {
    "quad_circle": quad_circle,
    "quad_ellipse": quad_ellipse,
    "quad_rotated_ellipse": quad_rotated_ellipse,
    "rosenbrock": rosenbrock,
    "linear": linear,
    "smooth_triangle": smooth_triangle
}

#create a directory to save the plots
output_dir = "report_plots"
os.makedirs(output_dir, exist_ok=True)

class TestUnconstrainedMin(unittest.TestCase):

    def test_all(self):
        for name, func in test_cases.items():
            with self.subTest(function=name):
                x0 = np.array([-1.0, 2.0]) if name == "rosenbrock" else np.array([1.0, 1.0])
                settings = plot_settings[name]

                # Run the optimization
                last_gd,*gd_results = minimize(func, x0, method='GD', max_iter=10000 if name == "rosenbrock" else 1000)
                if name != "linear":
                    last_nt, *nt_results = minimize(func, x0, method='NT', max_iter=100)
                #print the final line 
                print(f"{name} GD last iteration: {last_gd}")
                if name != "linear":
                    print(f"{name} NT last iteration: {last_nt}")
                plot_path = os.path.join(output_dir, f"{name}_paths.png")
                fval_path = os.path.join(output_dir, f"{name}_fval.png")

                #save the contour plot
                plot_contours_with_paths(
                    f=func,
                    xlim=settings["xlim"],
                    ylim=settings["ylim"],
                    levels=settings["levels"],
                    paths=[gd_results[3], nt_results[3]],
                    labels=["GD", "NT"],
                    title=f"{name.replace('_', ' ').title()}: Optimization Paths",
                    save_path=plot_path  # ✅ actually pass it
                )
                    
                plot_function_values(
                    f_paths=[gd_results[4], nt_results[4]],
                    labels=["GD", "NT"],
                    title=f"{name.replace('_', ' ').title()}: f(x) vs. Iteration",
                    save_path=fval_path
                )
                
                # Save the plots
                import matplotlib.pyplot as plt
                plt.savefig(plot_path)
                plt.close()

                plt.savefig(fval_path)
                plt.close()

if __name__ == "__main__":
    unittest.main()
import unittest
import sys
import os as os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.constrained_min import interior_point
from examples import (
    qp_objective, qp_ineq_constraints, qp_eq_constraints_mat, qp_eq_constraints_rhs, qp_x0,
    lp_objective, lp_ineq_constraints, lp_eq_constraints_mat, lp_eq_constraints_rhs, lp_x0
)
from src.utils import (
    plot_3d_feasible_region_with_path, 
    plot_2d_feasible_region_with_path,
    plot_objective_vs_iteration
)

class TestConstrainedMin(unittest.TestCase):

    def test_qp(self):
        """
        Test the quadratic programming example:
        min x² + y² + (z+1)²
        subject to: x + y + z = 1, x ≥ 0, y ≥ 0, z ≥ 0
        """
        print("\n" + "="*60)
        print("TESTING QUADRATIC PROGRAMMING (QP) EXAMPLE")
        print("="*60)
        print("Problem: min x² + y² + (z+1)²")
        print("Subject to: x + y + z = 1, x ≥ 0, y ≥ 0, z ≥ 0")
        print("Initial point:", qp_x0)
        
        # Run interior point method
        result = interior_point(
            func=qp_objective,
            ineq_constraints=qp_ineq_constraints,
            eq_constraints_mat=qp_eq_constraints_mat,
            eq_constraints_rhs=qp_eq_constraints_rhs,
            x0=qp_x0,
            mu=10.0,
            tol=1e-6,
            max_iter=100
        )
        
        x_final, f_final, success, outer_path, outer_f_vals = result
        
        # Print results
        self.assertTrue(success, "QP optimization should converge")
        print(f"\nQP Results:")
        print(f"Final solution: {x_final}")
        print(f"Final objective value: {f_final:.8f}")
        print(f"Converged: {success}")
        print(f"Number of outer iterations: {len(outer_f_vals)}")
        
        # Check constraints
        print(f"\nConstraint verification:")
        # Equality constraint: x + y + z = 1
        eq_violation = np.abs(np.sum(x_final) - 1.0)
        print(f"Equality constraint (x+y+z=1): sum = {np.sum(x_final):.8f}, violation = {eq_violation:.8f}")
        
        # Inequality constraints: x,y,z ≥ 0
        for i, val in enumerate(x_final):
            print(f"x[{i}] = {val:.8f} {'✓' if val >= -1e-6 else '✗'}")
        
        # Create output directory
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        
        # Plot 1: 3D feasible region and central path
        plot_3d_feasible_region_with_path(
            eq_constraints_mat=qp_eq_constraints_mat,
            eq_constraints_rhs=qp_eq_constraints_rhs,
            path=outer_path,
            title="QP: 3D Feasible Region and Central Path",
            save_path=os.path.join(out_dir, "qp_feasible_region_3d.png")
        )
        
        # Plot 2: objective value vs iteration
        plot_objective_vs_iteration(
            f_vals=outer_f_vals,
            title="QP: Objective Value vs Outer Iteration",
            save_path=os.path.join(out_dir, "qp_objective_vs_iteration.png")
        )
        
        print(f"QP plots saved to {out_dir}/")
        print("="*60)

    def test_lp(self):
        """
        Test the linear programming example:
        max[x + y] = min[-x - y]
        subject to: y ≥ -x + 1, y ≤ 1, x ≤ 2, y ≥ 0
        """
        print("\n" + "="*60)
        print("TESTING LINEAR PROGRAMMING (LP) EXAMPLE")
        print("="*60)
        print("Problem: max[x + y] = min[-x - y]")
        print("Subject to: y ≥ -x + 1, y ≤ 1, x ≤ 2, y ≥ 0")
        print("Initial point:", lp_x0)
        
        # Run interior point method
        result = interior_point(
            func=lp_objective,
            ineq_constraints=lp_ineq_constraints,
            eq_constraints_mat=lp_eq_constraints_mat,
            eq_constraints_rhs=lp_eq_constraints_rhs,
            x0=lp_x0,
            mu=10.0,
            tol=1e-6,
            max_iter=100
        )
        
        x_final, f_final, success, outer_path, outer_f_vals = result
        
        # Print results
        self.assertTrue(success, "LP optimization should converge")
        print(f"\nLP Results:")
        print(f"Final solution: {x_final}")
        print(f"Final objective value (min): {f_final:.8f}")
        print(f"Original objective value (max): {-f_final:.8f}")
        print(f"Converged: {success}")
        print(f"Number of outer iterations: {len(outer_f_vals)}")
        
        # Check constraints
        print(f"\nConstraint verification:")
        x, y = x_final[0], x_final[1]
        
        constraints = [
            ("y ≥ -x + 1", y >= -x + 1 - 1e-6, y - (-x + 1)),
            ("y ≤ 1", y <= 1 + 1e-6, y - 1),
            ("x ≤ 2", x <= 2 + 1e-6, x - 2),
            ("y ≥ 0", y >= -1e-6, -y)
        ]
        
        for constraint_name, satisfied, violation in constraints:
            status = "✓" if satisfied else "✗"
            print(f"{constraint_name}: violation = {violation:.8f} {status}")
        
        # Create output directory
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        
        # Plot 1: 2D feasible region and central path
        plot_2d_feasible_region_with_path(
            ineq_constraints=lp_ineq_constraints,
            xlim=(-0.5, 2.5),
            ylim=(-0.5, 1.5),
            path=outer_path,
            title="LP: 2D Feasible Region and Central Path",
            save_path=os.path.join(out_dir, "lp_feasible_region_2d.png")
        )
        
        # Plot 2: objective value vs iteration
        plot_objective_vs_iteration(
            f_vals=outer_f_vals,
            title="LP: Objective Value vs Outer Iteration",
            save_path=os.path.join(out_dir, "lp_objective_vs_iteration.png")
        )
        
        print(f"LP plots saved to {out_dir}/")
        print("="*60)

    def test_both(self):
        """
        Run both QP and LP tests
        """
        print("\n" + "="*60)
        print("RUNNING ALL CONSTRAINED OPTIMIZATION TESTS")
        print("="*60)
        
        self.test_qp()
        self.test_lp()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)

if __name__ == "__main__":
    # Run specific tests or all tests
    unittest.main()
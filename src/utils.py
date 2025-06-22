import matplotlib.pyplot as plt
import numpy as np
import os

def plot_contours_with_paths(f, xlim, ylim, paths=None, labels=None, title="Contour Plot", levels=30, save_path=None):
    """
    Plots 2D contour lines of the objective function and overlays optimization paths in color.

    - f: objective function (returns f_val, grad, hess)
    - xlim, ylim: (xmin, xmax), (ymin, ymax)
    - paths: list of lists of x points
    - labels: method names for legend
    """
    # Create a grid of points
    x1 = np.linspace(xlim[0], xlim[1], 400)
    x2 = np.linspace(ylim[0], ylim[1], 400)
    X1, X2 = np.meshgrid(x1, x2)

    # Compute the function values on the grid
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j],_,_ = f(np.array([X1[i, j], X2[i, j]]), need_hessian=False)

    # Plot the contours
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X1, X2, np.log1p(Z), levels=levels, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)

    #optionally print paths 
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple']
    markers = ['o', 's', 'D', '^', 'v', '*']
    if paths is not None:
        for i, path in enumerate(paths):
            path = np.vstack(path)
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.plot(path[:, 0], path[:, 1], color=color, marker=marker, label=labels[i] if labels else f'Method {i+1}')
    
    # plot
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    if paths:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    

def plot_function_values(f_paths, labels, title="Function Value per Iteration", save_path=None):
    """
    Plots f(x) vs iteration number for each method with colored lines.
    """
    #plot figure 
    plt.figure(figsize=(8, 6))
    colors = ['r', 'b', 'g', 'm', 'c', 'y']
    for i, f_vals in enumerate(f_paths):
        color = colors[i % len(colors)]
        plt.plot(f_vals, color=color, label=labels[i])
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_2d_feasible_region_with_path(ineq_constraints, xlim, ylim, path, title="2D Feasible Region and Central Path", save_path=None):
    """
    Plots the 2D feasible region defined by inequality constraints and the optimization path.
    
    Parameters:
    - ineq_constraints: list of constraint functions
    - xlim, ylim: plotting limits
    - path: optimization path (list of points)
    - title: plot title
    - save_path: path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a grid for constraint visualization
    x1 = np.linspace(xlim[0], xlim[1], 400)
    x2 = np.linspace(ylim[0], ylim[1], 400)
    X1, X2 = np.meshgrid(x1, x2)

    #find feasable region
    feasible_region = np.ones_like(X1, dtype=bool)

    for constraint in ineq_constraints:
        constraint_vals = np.zeros_like(X1)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                constraint_vals[i, j], _, _ = constraint(np.array([X1[i, j], X2[i, j]]), need_hessian=False)
        
        # Constraint satisfied where g(x) < 0
        feasible &= (constraint_vals < 0)

    # Plot the feasible region in green 
    ax.contourf(X1, X2, feasible.astype(int), levels=[0.5, 1.5], colors=['lightgreen'], alpha=0.3, label='Feasible Region')
    
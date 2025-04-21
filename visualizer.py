import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    """
    Visualizer handles the creation and display of scatter plots (1D, 2D, 3D)
    arranged in a specified grid layout.

    Parameters:
        layout (tuple): Grid layout defined as (nrows, ncols) for subplots.
        plot_specs (list): A list of dictionaries, each containing:
            - type (str): '1d', '2d', or '3d' indicating plot dimensionality.
            - clusters (array-like): Data to plot; shape must match the type.
            - kwargs (dict, optional): Additional settings for titles and labels.
    """
    def __init__(self, layout, plot_specs):
        """
        Initialize the Visualizer with a grid layout and plot specifications.

        This sets up internal attributes, validates the number of plots against layout,
        and ensures that each cluster array matches the declared plot dimensionality.
        """
        # Store layout and specifications
        self.layout = layout
        self.plot_specs = plot_specs
        self.nrows, self.ncols = layout
        self.n_plots = self.nrows * self.ncols

        # Ensure we don't exceed available subplot slots
        if len(plot_specs) > self.n_plots:
            raise ValueError(
                f"Layout {self.nrows}x{self.ncols} can only accommodate {self.n_plots} plots, "
                f"but {len(plot_specs)} plot specifications were provided."
            )

        # Prepare a default color cycle from matplotlib's Tab10 colormap
        self.colors = plt.get_cmap("tab10").colors

        # Validate each plot specification
        for spec in self.plot_specs:
            if "clusters" not in spec:
                raise ValueError("Each plot specification must include 'clusters'.")

            clusters = spec["clusters"]
            # Normalize single cluster to list for consistency
            if not isinstance(clusters, (list, tuple)):
                clusters = [clusters]

            ptype = spec.get("type")
            for cluster in clusters:
                # Validate array dimensions against declared plot type
                if ptype == "3d":
                    if cluster.ndim != 2 or cluster.shape[1] != 3:
                        raise ValueError(
                            "For '3d' plots, each cluster must be a 2D array with shape (N, 3)."
                        )
                elif ptype == "2d":
                    if cluster.ndim != 2 or cluster.shape[1] != 2:
                        raise ValueError(
                            "For '2d' plots, each cluster must be a 2D array with shape (N, 2)."
                        )
                elif ptype == "1d":
                    if cluster.ndim != 1:
                        raise ValueError(
                            "For '1d' plots, each cluster must be a 1D array."
                        )
                else:
                    raise ValueError("Plot type must be one of '3d', '2d', or '1d'.")

            # Update spec to ensure clusters is always a list
            spec["clusters"] = clusters

    def plot_scatter(self, ax, plot_type, clusters, **kwargs):
        """
        Render a scatter plot on the given Axes object.

        Parameters:
            ax (Axes): Matplotlib Axes or 3D Axes object to draw on.
            plot_type (str): '1d', '2d', or '3d'.
            clusters (list): List of numpy arrays containing cluster data.
            **kwargs: Optional title, xlabel, ylabel, zlabel overrides.
        """
        # Set defaults for titles and labels based on plot dimensionality
        if plot_type == "3d":
            default_title = "3D Scatter Plot"
            default_xlabel = "X"
            default_ylabel = "Y"
            default_zlabel = "Z"
        elif plot_type == "2d":
            default_title = "2D Scatter Plot"
            default_xlabel = "X"
            default_ylabel = "Y"
        else:  # '1d'
            default_title = "1D Scatter Plot"
            default_xlabel = "Value"

        # Iterate through clusters and plot each with its own color
        for i, cluster in enumerate(clusters):
            color = self.colors[i % len(self.colors)]
            # Label clusters when multiple exist
            label = f"Cluster {i+1}" if len(clusters) > 1 else None

            if plot_type == "3d":
                x, y, z = cluster[:, 0], cluster[:, 1], cluster[:, 2]
                ax.scatter(x, y, z, label=label, color=color, s=20)
            elif plot_type == "2d":
                x, y = cluster[:, 0], cluster[:, 1]
                ax.scatter(x, y, label=label, color=color, s=20)
            else:  # '1d'
                x = cluster
                y = np.zeros_like(cluster)
                ax.scatter(x, y, label=label, color=color, s=20)

        # Only show legend for more than one cluster
        if len(clusters) > 1:
            ax.legend()

        # Apply title and axis labels, using defaults if none provided
        ax.set_title(kwargs.get("title", default_title))
        ax.set_xlabel(kwargs.get("xlabel", default_xlabel))

        if plot_type in ("2d", "3d"):
            ax.set_ylabel(kwargs.get("ylabel", default_ylabel))
        if plot_type == "3d":
            ax.set_zlabel(kwargs.get("zlabel", default_zlabel))

        # Hide y-axis for 1D 'rug-style' scatter
        if plot_type == "1d":
            ax.get_yaxis().set_visible(False)

    def show(self):
        """
        Create a figure with the specified layout, plot all scatter plots, and display.

        Automatically selects 3D axes when needed and applies tight layout.
        """
        # Setup figure size proportional to grid dimensions
        fig = plt.figure(figsize=(self.ncols * 5, self.nrows * 5))

        # Add subplots and delegate plotting
        for i, spec in enumerate(self.plot_specs, start=1):
            ptype = spec.get("type")
            clusters = spec.get("clusters")
            kwargs = spec.get("kwargs", {})

            # Use 3D projection for 3D plots
            if ptype == "3d":
                ax = fig.add_subplot(self.nrows, self.ncols, i, projection="3d")
            else:
                ax = fig.add_subplot(self.nrows, self.ncols, i)

            # Plot each subplot
            self.plot_scatter(ax, ptype, clusters, **kwargs)

        # Adjust spacing and render
        plt.tight_layout()
        plt.show()

# Example usage                                            
if __name__ == "__main__":                            
    # Generate sample data for different dimensionalities  
    cluster1_3d = np.random.randn(100, 3) + np.array([2, 2, 2])
    cluster2_3d = np.random.randn(100, 3) + np.array([-2, -2, -2])
    clusters_3d = [cluster1_3d, cluster2_3d]

    x1 = np.linspace(0, 10, 100)                         
    y1 = np.sin(x1)
    x2 = np.linspace(0, 10, 100)
    y2 = np.cos(x2)
    clusters_2d = [
        np.column_stack((x1, y1)),
        np.column_stack((x2, y2))
    ]

    cluster1_1d = np.random.randn(200)
    cluster2_1d = np.random.randn(200) + 2
    clusters_1d = [cluster1_1d, cluster2_1d]

    plot_specs = [
        {"type": "3d", "clusters": clusters_3d, "kwargs": {"title": "3D Scatter Clusters"}},
        {"type": "2d", "clusters": clusters_2d, "kwargs": {"title": "2D Scatter Clusters", "xlabel": "X-axis", "ylabel": "Y-axis"}},
        {"type": "1d", "clusters": clusters_1d, "kwargs": {"title": "1D Scatter Plot", "xlabel": "Index"}}
    ]

    # Initialize and display the visualizations                  
    vis = Visualizer(layout=(2, 2), plot_specs=plot_specs)
    vis.show()

import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    """
    A utility for creating and displaying scatter plots in 1D, 2D, or 3D within a specified subplot grid.

    The dimensionality of each subplot is determined automatically based on the shape of provided data:
      - 1D: 1D array (N,)
      - 2D: 2D array with 2 columns (N, 2)
      - 3D: 2D array with 3 columns (N, 3)

    Attributes:
        layout (tuple[int, int]): Number of rows and columns for subplot grid.
        plot_specs (list[dict]): List of plot configurations, each containing:
            - 'clusters': single array or list of arrays to plot.
            - 'kwargs': optional dict for title, labels, etc.
            - 'type': inferred dimensionality ('1d'/'2d'/'3d').
        colors (tuple): Cycle of colors from the 'tab10' colormap.
    """
    def __init__(self, layout, plot_specs):
        # Unpack and validate grid dimensions
        self.layout = layout
        self.nrows, self.ncols = layout
        self.n_plots = self.nrows * self.ncols
        self.plot_specs = plot_specs

        # Prevent more plots than grid slots
        if len(self.plot_specs) > self.n_plots:
            raise ValueError(
                f"Grid {self.nrows}x{self.ncols} can only hold {self.n_plots} plots; "
                f"{len(self.plot_specs)} were provided."
            )

        # Prepare color cycle
        self.colors = plt.get_cmap('tab10').colors

        # Process and validate each plot specification
        for spec in self.plot_specs:
            if 'clusters' not in spec:
                raise KeyError("Plot spec missing 'clusters' key.")

            # Normalize to list of arrays
            clusters = spec['clusters']
            if not isinstance(clusters, (list, tuple)):
                clusters = [clusters]

            inferred_type = None
            validated_clusters = []
            for cluster in clusters:
                arr = np.asarray(cluster)
                # Infer plot dimensionality from array shape
                if arr.ndim == 1:
                    this_type = '1d'
                elif arr.ndim == 2 and arr.shape[1] == 2:
                    this_type = '2d'
                elif arr.ndim == 2 and arr.shape[1] == 3:
                    this_type = '3d'
                else:
                    raise ValueError(
                        f"Cannot infer plot type from shape {arr.shape}. "
                        "Expect 1D array or 2D array with 2 or 3 columns."
                    )

                # Ensure all clusters in one plot share the same dimensionality
                if inferred_type is None:
                    inferred_type = this_type
                elif inferred_type != this_type:
                    raise ValueError(
                        "All clusters in a single plot must have the same dimensionality."
                    )

                validated_clusters.append(arr)

            # Update spec with inferred type and validated arrays
            spec['type'] = inferred_type
            spec['clusters'] = validated_clusters

    def plot_scatter(self, ax, plot_type, clusters, **kwargs):
        """
        Render a scatter plot on the given Axes object.

        Args:
            ax (mpl.axes.Axes): Matplotlib Axes or 3D Axes to draw on.
            plot_type (str): One of '1d', '2d', or '3d'.
            clusters (list[np.ndarray]): Data arrays to plot.
            **kwargs: Optional overrides for 'title', 'xlabel', 'ylabel', 'zlabel'.
        """
        # Define default titles and labels per dimensionality
        if plot_type == '3d':
            default = dict(title='3D Scatter Plot', xlabel='X', ylabel='Y', zlabel='Z')
        elif plot_type == '2d':
            default = dict(title='2D Scatter Plot', xlabel='X', ylabel='Y')
        else:  # '1d'
            default = dict(title='1D Scatter Plot', xlabel='Value')

        # Plot each cluster with a distinct color
        for idx, cluster in enumerate(clusters):
            color = self.colors[idx % len(self.colors)]
            label = f"Cluster {idx+1}" if len(clusters) > 1 else None

            if plot_type == '3d':
                x, y, z = cluster.T
                ax.scatter(x, y, z, label=label, color=color, s=20)
            elif plot_type == '2d':
                x, y = cluster.T
                ax.scatter(x, y, label=label, color=color, s=20)
            else:
                # 1D: x-values versus zero baseline
                x = cluster
                y = np.zeros_like(cluster)
                ax.scatter(x, y, label=label, color=color, s=20)

        # Only show legend if multiple clusters exist
        if len(clusters) > 1:
            ax.legend()

        # Apply titles and axis labels
        ax.set_title(kwargs.get('title', default['title']))
        ax.set_xlabel(kwargs.get('xlabel', default['xlabel']))

        if plot_type in ('2d', '3d'):
            ax.set_ylabel(kwargs.get('ylabel', default['ylabel']))
        if plot_type == '3d':
            ax.set_zlabel(kwargs.get('zlabel', default['zlabel']))

        # Hide y-axis ticks for 1D plots
        if plot_type == '1d':
            ax.get_yaxis().set_visible(False)

    def show(self):
        """
        Create the Matplotlib figure, populate all subplots, and display.

        Automatically selects 3D projection when needed and applies a tight layout.
        """
        fig = plt.figure(figsize=(self.ncols * 5, self.nrows * 5))

        for idx, spec in enumerate(self.plot_specs, start=1):
            ptype = spec['type']
            clusters = spec['clusters']
            kwargs = spec.get('kwargs', {})

            # Choose 3D projection if required
            if ptype == '3d':
                ax = fig.add_subplot(self.nrows, self.ncols, idx, projection='3d')
            else:
                ax = fig.add_subplot(self.nrows, self.ncols, idx)

            # Delegate plotting to helper method
            self.plot_scatter(ax, ptype, clusters, **kwargs)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # --- Example Usage ---
    # 3D clusters: two Gaussian clouds
    cluster1_3d = np.random.randn(100, 3) + np.array([2, 2, 2])
    cluster2_3d = np.random.randn(100, 3) + np.array([-2, -2, -2])
    clusters_3d = [cluster1_3d, cluster2_3d]

    # 2D clusters: sine and cosine curves
    x = np.linspace(0, 10, 100)
    clusters_2d = [
        np.column_stack((x, np.sin(x))),
        np.column_stack((x, np.cos(x)))
    ]

    # 1D clusters: random samples
    clusters_1d = [np.random.randn(200), np.random.randn(200) + 3]

    # Define subplot grid and specifications
    plot_specs = [
        {'clusters': clusters_3d, 'kwargs': {'title': '3D Scatter'}},
        {'clusters': clusters_2d, 'kwargs': {'title': '2D Scatter'}},
        {'clusters': clusters_1d, 'kwargs': {'title': '1D Scatter'}}
    ]

    # Initialize and display visualizations in a 2x2 grid
    vis = Visualizer(layout=(2, 2), plot_specs=plot_specs)
    vis.show()
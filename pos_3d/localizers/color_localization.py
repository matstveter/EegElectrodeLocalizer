import numpy as np
import trimesh
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

from pos_3d.localizers.base_localizer import BaseLocalizer


class ColorLocalizer(BaseLocalizer):

    def __init__(self, rotation_matrix, orientation_dictionary, nasion, inion, lpa, rpa, origo,
                 multiple_thresholds=True, plot=False,
                 **kwargs):
        super().__init__(rotation_matrix=rotation_matrix, orientation_dictionary=orientation_dictionary,
                         nasion=nasion, inion=inion, lpa=lpa, rpa=rpa, origo=origo,
                         **kwargs)

        self.red_flag = False
        self._suggested_cluster_centers_rgb = None
        self._suggested_cluster_centers_hsv = None
        self._multiple_thresholds = multiple_thresholds
        self._plot = plot

    @property
    def suggested_cluster_centers_rgb(self):
        return self._suggested_cluster_centers_rgb

    @property
    def suggested_cluster_centers_hsv(self):
        return self._suggested_cluster_centers_hsv

    def localize_electrodes(self):
        """ Function that tries to estimate the cluster centers.

        Parameters
        ----------

        Returns
        -------
        None
        """

        if self._multiple_thresholds:
            self._suggested_cluster_centers_rgb = np.array(self._loop_segmentation(mesh=self._get_mesh()))
            self._suggested_cluster_centers_hsv = np.array(self._loop_segmentation(mesh=self._get_mesh(), use_rgb=False))

        else:
            rgb_mesh = self._color_segmentation(mesh=self._get_mesh())
            self._suggested_cluster_centers_rgb = np.array(self._cluster_colors(mesh=rgb_mesh))
            hsv_mesh = self._color_segmentation(mesh=self._get_mesh(), use_rgb=False)
            self._suggested_cluster_centers_hsv = np.array(self._cluster_colors(mesh=hsv_mesh))

        if len(self._suggested_cluster_centers_rgb) == 0 or len(self._suggested_cluster_centers_hsv) == 0:
            self.red_flag = True

        if self._plot:
            self.plot_solution(use_rgb=True)
            self.plot_solution(use_rgb=False)

    def _loop_segmentation(self, mesh, use_rgb=True, num_cluster_centers=135):
        """ Function that loops through different values of rgb and hsv to use different ranges of strict-ness in the
        values.

        Parameters
        ----------
        mesh: trimesh.Trimesh
            This is the mesh that should be segmented
        use_rgb: bool
            Use RGB or HSV, default=True

        Returns
        -------
        list
            cluster centers - 3D points
        """
        segmented_mesh = []
        for val in range(200, 50, -25):
            if use_rgb:
                seg_mesh = self._color_segmentation(mesh=mesh.copy(), use_rgb=use_rgb, min_value_rgb=val)
                segmented_mesh.append(seg_mesh)
            else:
                for sat in np.arange(0.1, 0.6, 0.1):
                    seg_mesh = self._color_segmentation(mesh=mesh.copy(), use_rgb=use_rgb, hsv_val_threshold=val,
                                                        hsv_sat_threshold=sat)
                    segmented_mesh.append(seg_mesh)

        most_cluster_centers = 0
        most_electrodes = None
        for seg in segmented_mesh:
            cluster_centers = self._cluster_colors(mesh=seg)

            if most_cluster_centers < len(cluster_centers) < num_cluster_centers:
                most_cluster_centers = len(cluster_centers)
                most_electrodes = cluster_centers
        return most_electrodes

    def _color_segmentation(self, mesh, use_rgb=True, min_value_rgb=75, hsv_sat_threshold=0.5, hsv_val_threshold=120):
        """ Segment the vertices based on either hsv, or rgb values.

        Parameters
        ----------
        mesh: trimesh.Trimesh
            Mesh to be segmented
        use_rgb:
            If True, segment based on rgb colors, else use HSV

        Returns
        -------
        trimesh.Trimesh
            mesh that has the segmented colors attached to it

        """

        def is_grey_white(vert, min_value, sat_threshold, value_threshold):
            if use_rgb:
                r, g, b = vert
                if min_value <= r <= 255 and min_value <= g <= 255 and min_value <= b <= 255:
                    return True
                else:
                    return False
            else:
                h, s, v = vert
                if s <= sat_threshold and v >= value_threshold:
                    return True
                else:
                    return False

        vertex_colors = self._get_colors(mesh=mesh, rgb=use_rgb)

        potential_vertex_colors = []
        indexes = []
        for i, vertex in enumerate(vertex_colors):
            if is_grey_white(vertex, min_value=min_value_rgb, sat_threshold=hsv_sat_threshold,
                             value_threshold=hsv_val_threshold):
                potential_vertex_colors.append(vertex)
                indexes.append(i)
        mask = np.ones(len(vertex_colors), dtype=bool)
        mask[indexes] = False

        # Set all other vertices to black
        vertex_colors[mask] = [0, 0, 0]

        new_mesh = mesh.copy()
        # Create a new visuals object with vertex colors
        new_visual = trimesh.visual.ColorVisuals(mesh=new_mesh, vertex_colors=vertex_colors)
        # Set the new visuals object to your mesh
        new_mesh.visual = new_visual
        # Now the mesh should have vertex colors
        return new_mesh

    @staticmethod
    def _cluster_colors(mesh, min_samples=3, max_samples=20):
        """ Function to cluster the colors and return the cluster centers.

        This function starts by extracting the vertices and the colors, and then finds the vertices that do not have
        the color white. Then calculates the distance between all of these vectors, use DBSCAN to cluster the vertices
        close "enough" to be labeled as a cluster. Then sort out clusters that are not correct, or with a limit
        exceeding the max-samples. Then find the center of each of the clusters and then return this values.

        Parameters
        ----------
        mesh: trimesh.Trimesh
            The mesh with the sorted colors, should be black colors for locations not of interest
        min_samples: int
            Minimum samples in a cluster
        max_samples: int:
            Maximum samples in a cluster

        Returns
        -------
        list
            All valid cluster centers
        """
        vertex_colors = mesh.visual.vertex_colors[:, :3]
        non_zero_indices = np.any(vertex_colors != [0, 0, 0], axis=1)
        filtered_vertices = mesh.vertices[non_zero_indices]

        distance_matrix = squareform(pdist(filtered_vertices, 'euclidean'))
        clustering = DBSCAN(eps=0.0075, min_samples=min_samples, metric="precomputed").fit(distance_matrix)

        labels = clustering.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_clusters = unique_labels[(unique_labels != -1) & (counts <= max_samples)]
        cluster_centers = [np.mean(filtered_vertices[labels == cluster_label], axis=0)
                           for cluster_label in valid_clusters]

        return cluster_centers

    def plot_solution(self, use_rgb=True, array=None):
        if array is None:
            if use_rgb and self._suggested_cluster_centers_rgb is not None:
                self._plot_mesh(rotation_matrix=self._rotation_matrix, points=self._suggested_cluster_centers_rgb)
            elif not use_rgb and self._suggested_cluster_centers_hsv is not None:
                self._plot_mesh(rotation_matrix=self._rotation_matrix, points=self._suggested_cluster_centers_hsv)
            else:
                # Do something
                raise NotImplementedError("")
        else:
            self._plot_mesh(rotation_matrix=self._rotation_matrix, points=array)


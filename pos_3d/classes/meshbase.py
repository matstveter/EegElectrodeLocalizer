from abc import ABC

import cv2
import numpy as np
import trimesh
import pyvista as pv
import matplotlib.colors as mcolors


class MeshBaseClass(ABC):
    def __init__(self, obj_file, jpg_file, subject_id, logger):
        self._point_transformation = None
        self._obj_file = obj_file
        self._jpg_file = jpg_file
        self._subject_id = subject_id
        self._logger = logger

        self._mesh = trimesh.load_mesh(file_obj=self._obj_file)
        self._mesh.process(validate=False)
        self._original_mesh = self._mesh.copy()

    @property
    def point_transformation(self):
        return self._point_transformation

    def transform_points(self, point_dict):
        if self._point_transformation is None:
            self._point_transformation = self._get_transformation(mesh=self._mesh)

        for key, point in point_dict.items():
            point_dict[key] = point - self._point_transformation
        return point_dict

    def _get_transformation(self, mesh=None):
        # Find the self._top point of the mesh -> self._top of head
        top_z_point = self._find_edge_points(mesh=mesh)[5]
        # Find the bounds of the mesh
        min_corner, max_corner = mesh.bounds
        # Create a new center point
        point_transform = (min_corner + max_corner) / 2
        # Specify that the all z-values should be negative, so that calculating distances is easier
        point_transform[2] = top_z_point[2]
        return point_transform

    def _find_edge_points(self, mesh=None):
        """ Function that returns the max and min values in all three directions, x, y, z. If mesh is None, use
        self._mesh

        Parameters
        ----------
        mesh: trimesh object, optional
            If none, use self._mesh else use the mesh sent to the function

        Returns
        -------
        min_x, max_x, min_y, max_y, min_z, max_z : np.ndarray
        """
        if mesh is None:
            mesh = self._mesh

        min_max_points = [(mesh.vertices[mesh.vertices[:, i].argmin()], mesh.vertices[mesh.vertices[:, i].argmax()])
                          for i in range(3)]
        x1, x2 = min_max_points[0]
        y1, y2 = min_max_points[1]
        z1, z2 = min_max_points[2]

        return np.array(x1), np.array(x2), np.array(y1), np.array(y2), np.array(z1), np.array(z2)

    def _find_middle_extrema(self, mesh=None, tolerance=0.05):
        if mesh is None:
            mesh = self._mesh

        vertices = mesh.vertices
        middle_x = np.mean([vertices[:, 0].max(), vertices[:, 0].min()])
        middle_y = np.mean([vertices[:, 1].max(), vertices[:, 1].min()])
        middle_z = np.mean([vertices[:, 2].max(), vertices[:, 2].min()])

        # Masks for points near middle x, y, z
        middle_x_mask = ((vertices[:, 1] > middle_y - tolerance) & (vertices[:, 1] < middle_y + tolerance) &
                         (vertices[:, 2] > middle_z - tolerance) & (vertices[:, 2] < middle_z + tolerance))
        middle_y_mask = ((vertices[:, 0] > middle_x - tolerance) & (vertices[:, 0] < middle_x + tolerance) &
                         (vertices[:, 2] > middle_z - tolerance) & (vertices[:, 2] < middle_z + tolerance))
        middle_z_mask = ((vertices[:, 0] > middle_x - tolerance) & (vertices[:, 0] < middle_x + tolerance) &
                         (vertices[:, 1] > middle_y - tolerance) & (vertices[:, 1] < middle_y + tolerance))

        # Filter vertices based on the masks
        middle_x_vertices = vertices[middle_x_mask]
        middle_y_vertices = vertices[middle_y_mask]
        middle_z_vertices = vertices[middle_z_mask]

        point_x1 = middle_x_vertices[middle_x_vertices[:, 0].argmin()]
        point_x2 = middle_x_vertices[middle_x_vertices[:, 0].argmax()]

        point_y1 = middle_y_vertices[middle_y_vertices[:, 1].argmin()]
        point_y2 = middle_y_vertices[middle_y_vertices[:, 1].argmax()]

        point_z1 = middle_z_vertices[middle_z_vertices[:, 2].argmin()]
        point_z2 = middle_z_vertices[middle_z_vertices[:, 2].argmax()]

        return point_x1, point_x2, point_y1, point_y2, point_z1, point_z2

    def _get_copy(self):
        """ Returns a copy of self._mesh using the trimesh library

        Returns
        -------
        trimesh.base.Trimesh
            A copy of the classes2 instance of self._mesh

        """
        return self._mesh.copy()

    def _get_colors(self, mesh=None, face_color=False, rgb=True, normalize=False, equalize=False, grayscale=False):
        """ Function to get the color of a mesh.

        Parameters
        ----------
        mesh: trimesh.base.Trimesh, optional
            mesh, if None, use the classes2 mesh
        face_color: bool, optional
            if True return the colors of the faces, if False, return the colors of the vertices
        rgb: bool, optional
            if True, return RGB colors, if False, return HSV
        normalize: bool, optional
            if normalize and rgb, divide all by 255, if hsv, only normalize V channel as the other are between 0 and 1
        equalize: bool, optional
            if True, equalize the histograms
        grayscale: bool, optional
            if True, convert to grayscale, if rgb=False, only return the V channel from HSV

        Returns
        -------
        np.ndarray:
            Color array with shape (N, 3), where N is either the number of vertices or the number of faces.
            Each row represents an RGB color associated with a vertex or a face.

        """
        # If mesh is None use the classes2 mesh
        if mesh is None:
            color_mesh = self._mesh.copy()
        else:
            color_mesh = mesh.copy()

        # Transform the mesh object to give color information instead, important to use copy, because this
        # destroys mapping
        color_mesh.visual = color_mesh.visual.to_color()

        # If face_color is true, return the color of the meshes faces, or the vertex colors
        if face_color:
            rgb_colors = color_mesh.visual.face_colors[:, :3]
        else:
            rgb_colors = color_mesh.visual.vertex_colors[:, :3]

        if grayscale:
            if rgb:
                grey_colors = 0.299 * rgb_colors[:, 0] + 0.587 * rgb_colors[:, 1] + 0.114 * rgb_colors[:, 2]
                return grey_colors
            else:
                hsv = mcolors.rgb_to_hsv(rgb_colors)
                v_channel = hsv[:, :, 2]
                return v_channel

        # If we want to equalize the histograms
        if equalize:
            equalized_colors = np.zeros_like(rgb_colors)
            for i in range(3):
                equalized_channel = cv2.equalizeHist(rgb_colors[:, i])
                equalized_colors[:, i] = np.squeeze(equalized_channel)
            rgb_colors = equalized_colors

        # If rgb, return rgb colors
        if rgb:
            if normalize:
                rgb_colors = rgb_colors / 255.0
            return rgb_colors
        else:
            hsv = mcolors.rgb_to_hsv(rgb_colors)
            if normalize:
                hsv[:, 2] = hsv[:, 2] / 255.0
            return hsv

    @staticmethod
    def _get_rotation_matrix_by_degrees(axis, degrees):
        """
        Generate a rotation matrix for a specified axis and angle in degrees.

        Parameters:
        -----------
        axis : str
            The axis of rotation ('x', 'y', or 'z').
        degrees : float
            The angle of rotation in degrees.

        Returns:
        --------
        numpy.ndarray
            A 3x3 rotation matrix that represents the rotation about the specified axis by the given angle.
        """
        theta = np.radians(degrees)  # Convert degrees to radians
        if axis == 'x':
            return np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        elif axis == 'y':
            return np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif axis == 'z':
            return np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("Invalid axis specified.")

    def _get_distances_of_vertices_in_same_plane(self, point, axis, tolerance=0.01, mesh=None, return_vertices=False):
        """
        Get vertices in the same plane as a given point along a specified axis and calculate
        the minimum and maximum distances between these vertices.

        Parameters:
        -----------
        point : array-like
            A point represented as [x, y, z] coordinates.
        axis : int
            The axis (0 for x, 1 for y, 2 for z) along which to check for vertices in the same plane.
        tolerance : float, optional
            Tolerance level for considering vertices in the same plane (default is 0.01).
        mesh : your_mesh_type, optional
            An optional mesh object. If not provided, it uses the internal mesh (_mesh) of the class.
        return_vertices: bool
            If this is set to True, also return all the vertices 3D positions

        Returns:
        --------
            max_distance: float
                representing the maximum distance between a pair of points of the vertices in the same plane
            selected_vertices: np.ndarray[3D]
                Optional if return_vertices is true, also return all vertices within the same plane

        """
        if mesh is None:
            mesh = self._mesh

        selected_vertices = mesh.vertices[np.isclose(mesh.vertices[:, axis], point[axis], atol=tolerance)]

        if return_vertices:
            return selected_vertices

        # Calculate distances between all pairs of selected vertices
        pairwise_distances = np.linalg.norm(selected_vertices[:, None] - selected_vertices, axis=-1)

        # Find the maximum distances
        max_distance = np.max(pairwise_distances[np.triu_indices(len(selected_vertices), k=1)])

        return max_distance

    @staticmethod
    def _get_max_min_point_from_vertices(vertices, axis):
        """
        Get the minimum and maximum points along a specified axis from a set of vertices.

        Parameters:
        -----------
        vertices : numpy.ndarray
            An array of vertices where each row represents a 3D point [x, y, z].
        axis : int
            The axis (0 for x, 1 for y, 2 for z) along which to find the minimum and maximum points.

        Returns:
        --------
        tuple
            A tuple containing two elements:
            - min_point (numpy.ndarray):
                The 3D point with the minimum value along the specified axis.
            - max_point (numpy.ndarray):
                The 3D point with the maximum value along the specified axis.
        """
        return vertices[np.argmin(vertices[:, axis])], vertices[np.argmax(vertices[:, axis])]

    def _plot_mesh(self, rotation_matrix, points=None, labels=None, colors=None, title=None):
        """
        Create a 3D plot with PyVista.

        Parameters:
            points (list of numpy arrays):
                An array of 3D points represented as spheres in the plot.
            labels (list, optional):
                Labels associated with each point in points. Displayed near the spheres.
            colors (list, optional):
                Optional colors for the spheres. If not provided, spheres are red by default.
            title (str, optional):
                An optional title for the plot. Defaults to "Subject ID: {self._subject_id}" if None.
        """
        # This is most likely a single point
        if len(np.array(points).shape) == 1:
            points = [points]

        # Open a plot
        plotter = pv.Plotter(off_screen=False)
        # Read the mesh and texture using the pyvista library
        mesh = pv.read(self._obj_file)
        texture = pv.read_texture(self._jpg_file)
        # Transform the mesh according to the rotation matrix
        mesh.points = np.dot(mesh.points, rotation_matrix.T)

        if self._point_transformation is not None:
            # Transform the mesh so that the coordinate system matches the mean of the head
            mesh.points = mesh.points - self._point_transformation

        if points is not None:
            for i, p in enumerate(points):
                # Add a sphere point for each of the points
                sphere = pv.Sphere(radius=0.005, center=p)
                # Set colors of the spheres, if None, use red
                if colors is None:
                    plotter.add_mesh(sphere, color="r")
                else:
                    plotter.add_mesh(sphere, color=colors[i])

        if labels is not None:
            # Add labels to the points that are created from the spheres
            plotter.add_point_labels(points=points, labels=labels, font_size=10, always_visible=True, shape_color="w")

        # Plot the mesh
        plotter.add_mesh(mesh, texture=texture)

        # Add title, set to subject id if it is None
        if title is None:
            plotter.add_text(text=f"Subject ID: {self._subject_id}", font_size=15)
        else:
            plotter.add_text(text=title, font_size=15)

        # Show the plot
        plotter.show()

    def get_rotated_and_transformed_mesh(self, rotation_matrix, orientation_dictionary=None):
        """ Rotate the self. mesh instance using the rotation matrix

        Parameters
        ----------
        rotation_matrix (3x3): np.ndarray
            rotation matrix

        Returns
        -------
        trimesh.Mesh

        """
        self._mesh.vertices = self._mesh.vertices.dot(rotation_matrix.T)
        points = self._get_transformation(mesh=self._mesh)
        self._mesh.apply_translation(-points)
        self._point_transformation = points

        if orientation_dictionary is not None:
            return self.transform_points(orientation_dictionary.copy())

    def _find_index(self, point, mesh=None, epsilon=1e-6):
        """
        Find the index of a vertex closest to the given point within a specified epsilon.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh object containing the vertices.
        point : numpy.ndarray
            The 3D point (x, y, z) to search for.
        epsilon : float
            The maximum allowable distance for a vertex to be considered "equal" to `point`.

        Returns
        -------
        int or None
            The index of the closest vertex within epsilon. If no such vertex exists, returns None.
        """
        if mesh is None:
            mesh = self._get_copy()
        distances = np.linalg.norm(mesh.vertices - point, axis=1)

        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < epsilon:
            return min_distance_idx
        else:
            return None

    def _find_vertices_with_color(self, mesh, point, radius, target_color=None):
        # Find vertices within the radius
        if target_color is None:
            target_color = [255, 255, 255]
        distances = np.linalg.norm(mesh.vertices - np.array(point), axis=1)
        within_radius_indices = np.where(distances < radius)[0]

        if len(within_radius_indices) == 0:
            # No vertices within this range...
            return None

        # Get colors, calculate distances, find the smallest
        colors = self._get_colors(mesh=mesh)[within_radius_indices]
        color_dists = np.linalg.norm(colors - np.array(target_color), axis=1)
        closest_index = np.argmin(color_dists)
        return mesh.vertices[within_radius_indices[closest_index]]

    def _find_center_of_point(self, mesh, point, radius=0.01, color_tolerance=60):
        """
        Find the center of a cluster of vertices around a given point within a mesh.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Mesh object containing vertex and color information.
        point : numpy.ndarray
            1D array specifying the x, y, z coordinates of the point around which to find the center.
        radius : float, optional
            Distance from the point within which to consider other vertices (default is 0.01).
        color_tolerance : int, optional
            Color distance tolerance for considering vertices as similar (default is 60).

        Returns
        -------
        numpy.ndarray
            1D array specifying the x, y, z coordinates of the center of the cluster of vertices.

        Notes
        -----
        The function uses Euclidean distance for both spatial distance and color difference.
        """

        # Get all vertex coordinates from the mesh
        vertices = np.array(mesh.vertices)

        # Get color information of vertices
        color = self._get_colors(mesh=mesh)

        # Compute the distance of each vertex color from white
        dist_from_white = np.linalg.norm(color - np.array([255, 255, 255]), axis=1)

        # Find the index of the given point in the mesh
        index_of_point = self._find_index(mesh=mesh, point=point)

        # Get the color distance of the given point from white
        cur_col = dist_from_white[index_of_point]

        # Calculate the Euclidean distance from the given point to all other points
        distances = np.linalg.norm(vertices - point, axis=1)

        # Get vertices within the specified radius from the point
        potential_vertices = vertices[distances < radius]

        potentials = []

        # Iterate through potential vertices to check color similarity
        for p in potential_vertices:
            ind = self._find_index(mesh=mesh, point=p)

            # Check if the color distance of the vertex is within the allowed tolerance
            if dist_from_white[ind] <= (cur_col + color_tolerance):
                potentials.append(p)

        # Calculate and return the mean position, serving as the cluster center
        return np.mean(np.array(potentials), axis=0)

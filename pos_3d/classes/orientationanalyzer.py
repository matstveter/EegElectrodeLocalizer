from matplotlib import pyplot as plt
import numpy as np

from pos_3d.classes.meshbase import MeshBaseClass
from pos_3d.utils.helper_functions import compare_hist, get_distance_between_points


class MeshOrientation(MeshBaseClass):
    def __init__(self, obj_file, subject_id, jpg_file, logger):
        super().__init__(obj_file=obj_file, jpg_file=jpg_file, subject_id=subject_id, logger=logger)
        self._detected_rotation_matrix = None
        self._top_point, self._bottom_point = None, None
        self._face, self._back, self._right, self._left = None, None, None, None
        self._point_dict = None

        # ASSUMPTIONS
        # This assumption is from the fact that we are using the same scanner and in the same position
        self._assumed_rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        # This will be the index of the array returned from middle_extrema
        self._assumed_face_direction = 2

    @property
    def top_point(self):
        return self._top_point

    @property
    def face(self):
        return self._face

    @property
    def back(self):
        return self._back

    @property
    def right(self):
        return self._right

    @property
    def left(self):
        return self._left

    @property
    def detected_rotation_matrix(self):
        return self._detected_rotation_matrix

    @property
    def point_dict(self):
        return self._point_dict

    def detect_orientation(self):
        """ Function that tries to calculate the rotation of the mesh, and sets the detected_rotation_matrix value.

        Calls two functions, each tasked with estimating the direction of the head and then return the rotation matrix
        which will transform the mesh so that the head is upright the z-axis.

        Returns
        -------

        """
        # Get a copy of the original mesh
        mesh = self._get_copy()
        # Calculate rotation matrix from the longest distance between vertices in the same plane
        rotation_matrix_suggestion1 = self._distances_of_vertices_in_same_plane()
        # Calculate rotation matrix from body distances and measurements
        rotation_matrix_suggestion2 = self._estimate_body_measurements(indentation_amount=0.08)

        # If the second method returns None, try with decreasing values, as the scan might be smaller
        if rotation_matrix_suggestion2 is None:
            ident_start = 0.08
            decrease = 0.02
            while rotation_matrix_suggestion2 is None:
                ident_start -= decrease
                rotation_matrix_suggestion2 = self._estimate_body_measurements(indentation_amount=ident_start)

        # If both rotation matrices returns the same
        if np.array_equal(rotation_matrix_suggestion1, rotation_matrix_suggestion2):
            if np.array_equal(rotation_matrix_suggestion1, self._assumed_rotation_matrix):
                self._logger.info("The detected orientation matched the assumed direction.")
                self._detected_rotation_matrix = rotation_matrix_suggestion1
            else:
                self._logger.warning("Both methods verifying direction are agreeing, but is not in the same direction "
                                     "as the initial assumption")
                self._detected_rotation_matrix = rotation_matrix_suggestion1
                # todo Potentially set as assumption?
        else:
            if (not np.array_equal(rotation_matrix_suggestion1, self._assumed_rotation_matrix) and
                    not np.array_equal(rotation_matrix_suggestion2, self._assumed_rotation_matrix)):
                self._logger.critical(f"Noisy scan, can not detect the orientation, using the assumed direction of the "
                                      f"scan!")
                print(f"Noisy scan, can find the direction: {self._subject_id}")
                self._detected_rotation_matrix = self._assumed_rotation_matrix
            elif (np.array_equal(rotation_matrix_suggestion1, self._assumed_rotation_matrix) and
                  (np.sum(mesh.extents) < 0.7)):

                self._logger.info("The bounding box around the scan is small and the body measurement method gave "
                                  "different value, indicating that the scan is only a head!")
                self._detected_rotation_matrix = rotation_matrix_suggestion1
            elif (np.array_equal(rotation_matrix_suggestion1, self._assumed_rotation_matrix) or
                  np.array_equal(rotation_matrix_suggestion2, self._assumed_rotation_matrix)):
                self._logger.info("One of the method agrees with the prior assumption!")
                self._detected_rotation_matrix = self._assumed_rotation_matrix
            else:
                self._logger("The two methods are not agreeing with each other or with the assumption, set to the the"
                             "second method, as the assumed rotation matrix is already defined in the class")
                self._detected_rotation_matrix = rotation_matrix_suggestion2

        # self.visualize_rotation(rotation_matrix=self._detected_rotation_matrix)

        # Rotate the original mesh
        self._mesh.vertices = self._mesh.vertices.dot(self._detected_rotation_matrix.T)
        # Get the top and bottom point
        _, _, _, _, self._bottom_point, self._top_point = self._find_middle_extrema()
        # Detect face, back, right and left, and set the values in the class to the suggestions
        self.detect_spatial_locations()
        self._verify_orientation()
        self._create_point_dictionary()

    def _verify_orientation(self):

        if np.argmax(self._mesh.extents) == 1 and (np.sum(self._mesh.extents) > 0.90):
            self._logger.warning("The depth of the subject is not smaller than the height or the width, suspecting"
                                 "something wrong or a smaller/wierd scan")
            print(f"Mesh Extents depths is not smallest: {self._mesh.extents}")
            self._mesh.show()

        if not self._get_spatial_measurements()[0]:
            self._logger.warning("Probability that the rotation is off is high!")
            print("Rotation is off!")

    def detect_spatial_locations(self):
        def get_potential_values(histo):
            """ This inner function returns the potential values for the location of the face based on histograms.

            This function receives a list of np.histograms. It then compares the different histograms, and then
            calculates the np. minimum value which should represent the most dissimilar histograms and the return
            the potential values based on the histograms that were compared.

            Parameters
            ----------
            histo: list
                Instance of numpy histograms from either a rgb array or a hsv array

            Returns
            -------
                list:
                    values which is the sides that are most dissimilar
            """
            hist_comparison = []
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    hist_comparison.append(compare_hist(hist1=histo[i], hist2=histo[j]))
            suggestion = np.argmin(hist_comparison)
            if suggestion == 0:
                return [0, 1]
            elif suggestion == 1:
                return [0, 2]
            elif suggestion == 2:
                return [0, 3]
            elif suggestion == 3:
                return [1, 2]
            elif suggestion == 4:
                return [1, 3]
            elif suggestion == 5:
                return [2, 3]
            else:
                raise ValueError("Unrecognized suggestion")

        # Get initial guess based on distances alone.
        distance, fb1, fb2, s1, s2 = self._get_spatial_measurements(from_top=0.10)
        x_min, x_max, y_min, y_max = self._find_middle_extrema()[:-2]

        spatial_orientation_pairs = [fb1, fb2, s1, s2]
        extremes_pair = [x_min, x_max, y_min, y_max]

        # Get copy of the mesh
        mesh = self._get_copy()
        offset_value = 0.05

        # Simplify the problem my removing parts of the scan
        mask = ((mesh.vertices[:, 2] < self._top_point[2] - offset_value) &
                (mesh.vertices[:, 2] > self._bottom_point[2] + offset_value))
        mesh.update_vertices(mask)

        # This means that the offset value was too great and the number of vertices is empty, trying without the latest
        # mask
        if mesh.vertices.shape[0] == 0:
            mesh = self._get_copy()
            self._logger.info("Skipping mask when estimating direction, due to smaller mesh. Might be because only the"
                              "head is included in the scan and not the upper body...")

        # Creates masks for the 4 sides of the head based on the extreme values in each of the axis max min
        x_min_mask = (mesh.vertices[:, 0] >= x_min[0]) & (mesh.vertices[:, 0] < mesh.centroid[0] - offset_value)
        x_max_mask = (mesh.vertices[:, 0] <= x_max[0]) & (mesh.vertices[:, 0] > mesh.centroid[0] + offset_value)
        y_min_mask = (mesh.vertices[:, 1] >= y_min[1]) & (mesh.vertices[:, 1] < mesh.centroid[1] - offset_value)
        y_max_mask = (mesh.vertices[:, 1] <= y_max[1]) & (mesh.vertices[:, 1] > mesh.centroid[1] + offset_value)

        # Save the masks in a list
        masks = [x_min_mask, x_max_mask, y_min_mask, y_max_mask]

        # Create empty lists for storing values
        mean_normals, var_normals, hist_hsv, hist_rgb = [], [], [], []
        # Empty list for storing the created meshes
        meshes = []
        for m in masks:
            # Get a new copy
            temp_mesh = mesh.copy()
            # Update the vertices according to the mask in the masks list
            temp_mesh.update_vertices(m)
            # Append the actual mesh to the empty list
            meshes.append(temp_mesh)

            # Calculate the mean of the vertex normals and store in list
            mean_normals.append(np.mean(temp_mesh.vertex_normals))
            # Calculate the variance of the vertex normals and store in list
            var_normals.append(np.var(temp_mesh.vertex_normals))
            # Look at the Hue value in HSV colors, and calculate the histogram of that color and append to list
            hist_hsv.append(np.histogram(self._get_colors(mesh=temp_mesh, rgb=False)[:, 0], bins=360, range=(0, 1))[0])
            # Look at the RGB colors of the vertices and store in list
            hist_rgb.append(np.histogram(self._get_colors(mesh=temp_mesh), bins=256, range=(0, 255))[0])

        # Use the function, get_potential_values to extract the potential face location based on the histograms
        rgb_dissimilar_axis = get_potential_values(histo=hist_rgb)
        hsv_dissimilar_axis = get_potential_values(histo=hist_hsv)

        # From the spatial distances calculated earlier, transform this to fit the extreme points
        temp_face_axis = []
        for p in extremes_pair:
            temp_dist = []
            for p2 in spatial_orientation_pairs:
                temp_dist.append(get_distance_between_points(p1=p, p2=p2))
            temp_face_axis.append(np.argmin(temp_dist))

        # Get the potential indexes from the array
        face_values = temp_face_axis[0:2]
        side_values = temp_face_axis[2:4]

        # Calculate the argmin of the normals closest to zero and the argmax of the variance
        suggested_face_mean_normal = np.argmin(np.abs(mean_normals))
        suggested_face_variance = np.argmax(var_normals)

        # Compare the methods to the assumption and save and store the face, back, right and left sides as point
        if (distance and np.array_equal(face_values, hsv_dissimilar_axis)) or (
                distance and np.array_equal(face_values, rgb_dissimilar_axis)):
            # This means that the distances can be trusted based on the assumptions where the face is and that it
            # matches with colors
            if suggested_face_variance in face_values and suggested_face_variance == 2:
                self._face = extremes_pair[suggested_face_variance]
                self._back = extremes_pair[suggested_face_variance + 1]
                self._right = extremes_pair[side_values[0]]
                self._left = extremes_pair[side_values[1]]
            elif suggested_face_mean_normal in face_values and suggested_face_mean_normal == 2:
                self._face = extremes_pair[suggested_face_mean_normal]
                self._back = extremes_pair[suggested_face_mean_normal + 1]
                self._right = extremes_pair[side_values[0]]
                self._left = extremes_pair[side_values[1]]
            else:
                self._logger.info("Set to assumed direction of face")
                self._face = extremes_pair[self._assumed_face_direction]
                self._back = extremes_pair[self._assumed_face_direction + 1]
                self._right = extremes_pair[side_values[0]]
                self._left = extremes_pair[side_values[1]]
        else:
            self._logger.warning("Face assumed to without the scope of the assumed face orientation, suggests"
                                 "verifying this manually.")
            if (np.array_equal(face_values, hsv_dissimilar_axis) and np.array_equal(face_values, rgb_dissimilar_axis)
                    and suggested_face_variance in face_values and suggested_face_mean_normal in face_values):
                self._logger.warning("Distances failed, but 4 methods suggesting face based on colors, and the "
                                     "curvature are agreeing on the position of the face, trusting the methods over "
                                     "the assumption")
                if suggested_face_variance == 3:
                    self._face = extremes_pair[suggested_face_variance]
                    self._back = extremes_pair[suggested_face_variance - 1]
                    self._right = extremes_pair[side_values[1]]
                    self._left = extremes_pair[side_values[0]]
                else:
                    self._face = extremes_pair[suggested_face_variance]
                    self._back = extremes_pair[suggested_face_variance + 1]
                    self._right = extremes_pair[side_values[0]]
                    self._left = extremes_pair[side_values[1]]
            else:
                self._logger.warning("Distance, and methods all lie outside the assumed face direction, setting the "
                                     "assumed face direction to the initial assumption. Suggest to manually verify!")
                self._face = extremes_pair[self._assumed_face_direction]
                self._back = extremes_pair[self._assumed_face_direction + 1]
                self._right = extremes_pair[0]
                self._left = extremes_pair[1]

    def _distances_of_vertices_in_same_plane(self):
        """ Function that calculates the distances among vertices in the same plane.

        The assumption is that when a 3D image is captured, a bounding box must be defined. This bounding box leaves
        straight edges in the torso of the subject. After the vertices in the same plane is calculated we return the
        plane of where the distances among the vertices in the same plane were greatest.

        Notes:
        This function is not robust if there are a lot of noise surrounding the mesh. Because this function is based on
        the bounding box of the mesh.

        Returns
        -------
            np.ndarray:
                the output _get_rotation_matrix function which returns the rotation matrix
        """
        mesh_bounds = self._find_edge_points()
        axis = [0, 0, 1, 1, 2, 2]
        vertices_in_same_plane = \
            [self._get_distances_of_vertices_in_same_plane(point=e, axis=axis[i]) for i, e in enumerate(mesh_bounds)]

        # If the largest distance is in direction 0 meaning x_min, then it should be returned as 0
        return self._get_rotation_matrix(bottom=np.argmax(vertices_in_same_plane))

    def _estimate_body_measurements(self, indentation_amount=0.08):
        """ Estimates body measurements, calculate the min-max distance in 2/3 axis in all three axis. The minimum
        distance should be the head.

        The assumption is that the smallest distance should be in the x-y min-max distances in the head.

        Parameters
        ----------
        indentation_amount: float
            How much to go towards the center from each fo the extreme points in cm

        Returns
        -------
            np.ndarray:
                Calls function for getting the rotation matrix, with the estimated head position
        """

        mesh_bounds = self._find_middle_extrema()
        axis = [0, 0, 1, 1, 2, 2]

        # How far should we go along each axis before calculating the edge points
        indentation_towards_center = indentation_amount
        direction = [indentation_towards_center, -indentation_towards_center]

        # List for storing distances
        distances = []

        for i in range(len(mesh_bounds)):
            if i % 2 == 0:
                direct = direction[0]
            else:
                direct = direction[1]

            # Get one of the point in the bounds, and go towards the center of the mesh by distance
            temp_point = mesh_bounds[i]
            temp_point[axis[i]] += direct

            # Filter vertices which is in that plane
            filtered_vertices = self._get_distances_of_vertices_in_same_plane(point=temp_point,
                                                                              axis=axis[i],
                                                                              tolerance=0.02,
                                                                              return_vertices=True)
            if filtered_vertices.shape[0] == 0:
                self._logger.warning("Estimating Direction: The indentation number were too large, leading to "
                                     "masks creating empty arrays, trying with smaller amounts!")
                return None

            # Choose which axis to focus on
            filter_axis = np.unique([x for x in axis if x != axis[i]])

            # Find extremes at that location in the mesh, and the distance between the extremes
            p1_min, p1_max = self._get_max_min_point_from_vertices(vertices=filtered_vertices, axis=filter_axis[0])
            p1_distance = get_distance_between_points(p1=p1_min, p2=p1_max)
            p2_min, p2_max = self._get_max_min_point_from_vertices(vertices=filtered_vertices, axis=filter_axis[1])
            p2_distance = get_distance_between_points(p1=p2_min, p2=p2_max)

            distances.append(p1_distance + p2_distance)

        # The smallest distance between the extremes 15 cm towards the center, should be the head
        return self._get_rotation_matrix(head=np.argmin(distances))

    def _get_rotation_matrix(self, axis=None, head=None, bottom=None):
        """
        Get a rotational matrix corresponding to a specified axis.

        Parameters:
        -----------
        axis : str or int
            The axis for which to obtain the rotational matrix.
            Supported values: "xmin", "xmax", "ymin", "ymax", "zmin", or any other string (returns identity matrix).
            if xmin, this means that this is the values which should then be the "new" zmax

        Returns:
        --------
        numpy.ndarray
            A 3x3 rotational matrix corresponding to the specified axis.
        """
        if axis is None and head is None and bottom is None:
            raise ValueError("No axis is chosen!")

        if axis == "xmin" or head == 0 or bottom == 1:
            return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        elif axis == "xmax" or head == 1 or bottom == 0:
            return np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]])
        elif axis == "ymin" or head == 2 or bottom == 3:
            return np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]])
        elif axis == "ymax" or head == 3 or bottom == 2:
            return self._assumed_rotation_matrix
        elif axis == "zmin" or head == 4 or bottom == 5:
            return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        elif axis == "zmax" or head == 5 or bottom == 4:
            return np.identity(3)
        else:
            raise ValueError("Invalid axis specified!")

    def visualize_rotation(self, rotation_matrix):
        """
        Visualize the effect of a rotation on a mesh.

        Parameters:
        -----------
        rotation_matrix : numpy.ndarray
            A 3x3 rotation matrix representing the desired rotation.

        Returns:
        --------
        None

        """
        original_mesh = self._get_copy()
        rotated_mesh = self._get_copy()
        rotated_mesh.vertices = rotated_mesh.vertices.dot(rotation_matrix.T)

        fig = plt.figure(figsize=(12, 6))
        # Original mesh
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title('Original Mesh')
        ax1.plot_trisurf(original_mesh.vertices[:, 0], original_mesh.vertices[:, 1], original_mesh.faces,
                         original_mesh.vertices[:, 2])

        ax1.set_xlabel('X Label')
        ax1.set_ylabel('Y Label')
        ax1.set_zlabel('Z Label')
        # Rotated mesh
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title('Rotated Mesh')
        ax2.plot_trisurf(rotated_mesh.vertices[:, 0], rotated_mesh.vertices[:, 1], rotated_mesh.faces,
                         rotated_mesh.vertices[:, 2])

        ax2.set_xlabel('X Label')
        ax2.set_ylabel('Y Label')
        ax2.set_zlabel('Z Label')

        plt.show()

    def _get_spatial_measurements(self, from_top=0.10):
        # Go 8 cm from the top of the head
        head_slice_height = self._top_point[2] - from_top
        # Give a tolerance on the mesh.vertices z-coordinates
        cutoff_margin = 0.01
        # Create a mask that sorts out the vertices of interest
        temp_mask = (self._mesh.vertices[:, 2] > head_slice_height -
                     cutoff_margin) & (self._mesh.vertices[:, 2] < head_slice_height + cutoff_margin)
        # Get the vertices with the applied mask
        masked_vertices = self._mesh.vertices[temp_mask]

        # Find the max and min values in all directions
        point1 = masked_vertices[masked_vertices[:, 0].argmin()]
        point2 = masked_vertices[masked_vertices[:, 0].argmax()]
        point3 = masked_vertices[masked_vertices[:, 1].argmin()]
        point4 = masked_vertices[masked_vertices[:, 1].argmax()]

        # Calculate the middle x and y coordinates
        middle_12 = (point1[0] + point2[0]) / 2
        middle_34 = (point3[1] + point4[1]) / 2

        # Create a mask to find vertices near the middle y-coordinate
        tolerance = 0.01
        middle_y_mask = (masked_vertices[:, 1] > middle_34 - tolerance) & \
                        (masked_vertices[:, 1] < middle_34 + tolerance)
        middle_x_mask = (masked_vertices[:, 0] > middle_12 - tolerance) & \
                        (masked_vertices[:, 0] < middle_12 + tolerance)

        # Filter vertices using the middle_y_mask
        middle_y_vertices = masked_vertices[middle_y_mask]
        middle_x_vertices = masked_vertices[middle_x_mask]

        # Now find max and min x-values among these middle_y_vertices
        point1 = middle_y_vertices[middle_y_vertices[:, 0].argmin()]
        point2 = middle_y_vertices[middle_y_vertices[:, 0].argmax()]
        point3 = middle_x_vertices[middle_x_vertices[:, 1].argmin()]
        point4 = middle_x_vertices[middle_x_vertices[:, 1].argmax()]

        # Calculate the distance between the points in x direction and y direction
        p1p2 = get_distance_between_points(point1, point2)
        p3p4 = get_distance_between_points(point3, point4)

        if p1p2 > p3p4:
            return False, point1, point2, point3, point4
        else:
            return True, point3, point4, point1, point2

    def _create_point_dictionary(self):
        point_dict = {'top': self._top_point,
                      'bottom': self._bottom_point,
                      'face': self._face,
                      'back': self.back,
                      'right': self.right,
                      'left': self._left}

        self._point_dict = point_dict
        return point_dict

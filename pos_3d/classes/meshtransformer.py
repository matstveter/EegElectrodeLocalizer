from abc import ABC

import numpy as np

from pos_3d.classes.meshbase import MeshBaseClass
from pos_3d.utils.helper_functions import get_cutoff_axis


class MeshTransformer(MeshBaseClass, ABC):
    def __init__(self, rotation_matrix, orientation_dictionary, **kwargs):
        super().__init__(**kwargs)
        transformed_orientation_dictionary = (
            self.get_rotated_and_transformed_mesh(rotation_matrix=rotation_matrix,
                                                  orientation_dictionary=orientation_dictionary))
        self._point_transform = None
        self._head_mesh = None
        self._rotation_matrix = rotation_matrix
        self._orientation_dictionary = transformed_orientation_dictionary

        # https://www.researchgate.net/publication/264024487_Analysis_of_Human_Head_Shapes_in_the_United_States
        self._assumed_head_size = 0.20
        self._assumed_nasion_height_interval = self._assumed_head_size / 3

        # This will do a transformation of the points in the mesh
        self._head_mesh = self.get_head_mesh()

        # Init again, because the orientation dictionary will change
        self._top = transformed_orientation_dictionary['top']
        self._bottom = transformed_orientation_dictionary['bottom']
        self._face = transformed_orientation_dictionary['face']
        self._back = transformed_orientation_dictionary['back']
        self._right = transformed_orientation_dictionary['right']
        self._left = transformed_orientation_dictionary['left']

    def get_head_mesh(self):
        """
        Slice the upper portion of the mesh, assumed to represent the head, based on a predetermined height.

        Attributes
        ----------
        self._mesh : trimesh.Trimesh
            The mesh object to be sliced.
        self._top : tuple of float
            The coordinates of the self._top-most vertex in the mesh.
        self._assumed_head_height : float
            The height from the self._top vertex assumed to encompass the head.
        self._plot : bool
            Whether to plot the new mesh after slicing.

        Notes
        -----
        1. Create a copy of the original mesh.
        2. Identify the vertices and faces that are part of the "head" based on a predetermined height.
        3. Update the mesh vertices and faces based on the mask.
        4. Update all points of the mesh so that the points are set to more properly to xyz axis
        5. Plot the new mesh, if required.
        """
        if self._rotation_matrix is None:
            self._logger.warning("Head Extraction: The mesh has not been rotated, so unless you are certain that "
                                 "the self._top of the head is the maximum value in z direction as a "
                                 "standard, this will not work")

        # Get self._top and bottom points from dictionary and calculate distance
        dist_top_bottom = np.abs(self._orientation_dictionary['top'][2] - self._orientation_dictionary['bottom'][2])

        if dist_top_bottom < self._assumed_head_size:
            self._logger.info('Head height is less than the assumed distance, using self._top-bottom distance')
            head_cutoff = self._orientation_dictionary['top'][2] - dist_top_bottom
        else:
            head_cutoff = self._orientation_dictionary['top'][2] - self._assumed_head_size

        self._orientation_dictionary['new_bottom'] = head_cutoff
        # Create a copy of the original mesh
        new_mesh = self._mesh.copy()
        # Define the mask for vertices above the slicing plane
        mask = new_mesh.vertices[:, 2] >= head_cutoff
        # Define the mask for faces above the slicing plane
        face_avg_z = np.mean(new_mesh.vertices[new_mesh.faces][:, :, 2], axis=1)
        face_mask = face_avg_z >= head_cutoff

        # Update faces based on the face mask
        new_mesh.update_faces(face_mask)

        # Update vertices based on the mask
        new_mesh.update_vertices(mask)
        return new_mesh

    def _slice_quarter_from_top(self):
        mesh = self._head_mesh.copy()
        # Remove the 1/4 top of the head, the nasion cannot be there
        face_avg_z = np.mean(mesh.vertices[mesh.faces][:, :, 2], axis=1)
        face_mask = (face_avg_z < self._top[2] - self._assumed_nasion_height_interval)
        mask = (mesh.vertices[:, 2] < self._top[2] - self._assumed_nasion_height_interval)
        mesh.update_faces(face_mask)
        mesh.update_vertices(mask)
        return mesh

    def _get_face_slice(self, mesh, tolerance=0.02):
        # Go from the vertices centroid, towards the face, and only keep that, first update faces and then vertices
        axis = get_cutoff_axis(centroid=mesh.centroid, point=self._face)

        # For each face, look up its vertices and take the average along the specific axis
        average_face_points = np.max(mesh.vertices[mesh.faces][:, :, axis], axis=1)
        # Create a mask based on the average points
        face_mask = (average_face_points >= self._face[axis]) & \
                    (average_face_points <= (mesh.centroid[axis] - tolerance))
        mask = (mesh.vertices[:, axis] >= self._face[axis]) & (
                mesh.vertices[:, axis] <= (mesh.centroid[axis] - tolerance))
        mesh.update_faces(face_mask)
        mesh.update_vertices(mask)

        return mesh

    def _get_center_plus_tolerance(self, mesh, axis, tolerance=0.02):
        extremes = self._find_middle_extrema(mesh=self._head_mesh)

        # Get the middle of the chosen axis
        if axis == 0:
            middle = (extremes[0][0] + extremes[1][0]) / 2
        elif axis == 1:
            middle = (extremes[2][1] + extremes[3][1]) / 2
        elif axis == 2:
            middle = (extremes[4][2] + extremes[5][2]) / 2
        else:
            raise ValueError(f"Unrecognized value: {axis}")

        average_face = np.max(mesh.vertices[mesh.faces][:, :, axis], axis=1)
        face_mask = (average_face > middle - tolerance) & (average_face < middle + tolerance)
        mask = (mesh.vertices[:, axis] > middle - tolerance) & (mesh.vertices[:, axis] < middle + tolerance)
        mesh.update_faces(face_mask)
        mesh.update_vertices(mask)

        return mesh

    def _white_color_search(self, mesh, nasion, min_distance=0.01, max_candidates=10, white_cutoff_distance=200):
        """ This function finds potential candidates for nasion or rpa_lpa based on the sliced mesh and white color.


        Parameters
        ----------
        mesh: trimesh.Trimesh
            Sliced mesh, with only the parts of the mesh where the lpa and rpa can potentially be
        nasion: bool
            Changes the search criteria a little bit
        min_distance: float, optional
            defaults to 0.02, sets the minimum distance a candidate has to be away from already selected candidates
        max_candidates: int, optional
            max number of candidates that should be returned from this function
        white_cutoff_distance: int, optional
            if the vertex has less than this distance to white, it is a potential candidate

        Returns
        -------
            list, list:
            potential candidates, indexes of potential candidates
        """
        # Function values
        white_color = np.array([255, 255, 255])

        # Get the colors of the vertices in RGB
        vertices_color = self._get_colors(mesh=mesh)

        # Calculate the distance from each vertex to the color white using Euclidean distance
        distance_to_color_white = np.linalg.norm(vertices_color - white_color, axis=1)

        # Sort the indices based on closeness to white
        sorted_white_indices = np.argsort(distance_to_color_white)

        # init empty lists to store potential candidates
        candidates, candidate_color = [], []

        # Loop through the sorted white indices
        for idx in sorted_white_indices:
            # Get the actual 3D point of the current candidate
            candidate_point = mesh.vertices[idx]

            # Get the distance to the color white for the current candidate
            candidate_distance_to_white = distance_to_color_white[idx]

            # Calculate the number of neighbors which is relatively close to white
            candidate_white_ich_neighbors = len(list(filter(lambda x: x < white_cutoff_distance,
                                                            distance_to_color_white[mesh.vertex_neighbors[idx]])))

            # Check that the distance to white is less than 175, number of neighbors with white ich color is more than 1
            # and that the distance between this point and already selected candidates is more or equal to min_distance
            if nasion:
                if all(
                        candidate_distance_to_white < white_cutoff_distance
                        and candidate_white_ich_neighbors > 1
                        and np.linalg.norm(candidate_point[2] - point[2]) >= min_distance
                        for point in candidates
                ):
                    # Append the candidates
                    candidates.append(candidate_point)
                    candidate_color.append(candidate_distance_to_white)
            else:
                if all(
                        candidate_distance_to_white < white_cutoff_distance
                        and candidate_white_ich_neighbors > 1
                        and np.linalg.norm(candidate_point - point) >= min_distance
                        for point in candidates
                ):
                    # Append the candidates
                    candidates.append(candidate_point)
                    candidate_color.append(candidate_distance_to_white)

            # If the list contains the max number of candidates break the loop
            if len(candidates) == max_candidates:
                break
        return candidates, candidate_color

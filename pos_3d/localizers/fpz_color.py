import numpy as np
from pos_3d.localizers.color_localization import ColorLocalizer
from pos_3d.utils.mesh_helper_functions import  manual_point_selection


class FPzFinder(ColorLocalizer):

    def __init__(self, rotation_matrix, orientation_dictionary, nasion, inion, lpa, rpa, origo, template_dict,
                 multiple_thresholds=True, **kwargs):
        super().__init__(rotation_matrix=rotation_matrix, orientation_dictionary=orientation_dictionary,
                         nasion=nasion, inion=inion, lpa=lpa, rpa=rpa, origo=origo,
                         multiple_thresholds=multiple_thresholds,
                         **kwargs)
        self._fpz = None
        self._dist_fpz_nasion = np.linalg.norm(template_dict['Nz'] - template_dict['FPz'])

    @property
    def fpz(self):
        return self._fpz

    def _get_electrode_suggestions(self, use_rgb=True):
        """ Function to get suggested electrodes centers based on a midline mesh, and then sort out candidates that
        are possible candidates. This involves having roughly the same x-value as the nasion point, as well as not
        being the nasion point.

        Parameters
        ----------
        use_rgb: bool
            default True, use rgb or hsv

        Returns
        -------
        np.ndarray
            the potential cluster centers

        """
        # Get mesh only containing the midline and an offset
        midline_front_back = self._get_center_plus_tolerance(mesh=self._get_mesh(), axis=0, tolerance=0.025)
        cluster_centers = self._cluster_colors(mesh=self._color_segmentation(mesh=midline_front_back, use_rgb=use_rgb,
                                                                             min_value_rgb=50, hsv_sat_threshold=0.5,
                                                                             hsv_val_threshold=100))

        # Loop through and check that the x-axis roughly corresponds to the nasion x-axis and that distance from nasion
        # is above a threshold, to avoid having nasion as a candidate
        midline_centers = []
        for e in cluster_centers:
            if np.abs(self._nasion[0] - e[0]) < 0.005 and 0.075 > np.linalg.norm(self._nasion - e) > 0.02:
                midline_centers.append(e)

        return np.array(midline_centers)

    def _estimate_fpz(self):
        """
        Estimate the FPz point on a 3D mesh by casting a ray from a starting point behind the nasion.

        This method calculates the estimated FPz point on a 3D mesh by casting a ray from a starting point slightly
        behind the nasion in the direction of the positive x-axis and finding the first intersection point with the
         mesh.

        Returns
        -------
        numpy array or None
            The estimated FPz point as a 3D numpy array [x, y, z] if an intersection point is found.
            Returns None if no valid intersection point is found.

        Notes
        -----
        - The `_get_mesh()` method should be implemented to return the 3D mesh.
        - The `_nasion` and `_dist_fpz_nasion` attributes should be set with appropriate values before calling this
        method.
        """
        # Get the mesh (assuming you have a _get_mesh() method that returns a mesh)
        mesh = self._get_mesh()

        # Define the starting point for the ray as a point slightly behind the nasion
        start_point = self._nasion + [0, 0, self._dist_fpz_nasion]

        # Calculate directions for ray-tracing
        direction_to_centroid = mesh.centroid - start_point
        direction_away_from_centroid = -direction_to_centroid

        # Perform ray-tracing
        intersection_points = mesh.ray.intersects_location(
            ray_origins=[start_point, start_point],
            ray_directions=[direction_to_centroid, direction_away_from_centroid]
        )[0]  # Assuming this extracts the intersection points from the results

        # Check if there are any intersection points
        if len(intersection_points) > 0:
            # Calculate distances from the estimated position to each intersection point
            dists_to_estimated_point = [np.linalg.norm(start_point - e) for e in intersection_points]

            # Find the index of the minimum distance
            min_distance_index = np.argmin(dists_to_estimated_point)

            # Select the closest intersection point
            estimated_pos = intersection_points[min_distance_index]
        else:
            # If there are no intersection points, keep the original estimated position
            estimated_pos = start_point

        return estimated_pos

    def _get_fpz(self, arr):
        """ Function that calculates the distance from nasion to choose the FPz values

        Parameters
        ----------
        arr : list of numpy arrays
            List of points in 3D space.

        Returns
        -------
        numpy array
            The chosen FPz value based on distance from nasion.
        """
        if len(arr) > 1:
            dists = [np.linalg.norm(self._nasion - e) for e in arr]
            min_dist = min(dists)
            if min_dist < 0.02:
                # If the minimum distance is less than 0.02, choose the second minimum value
                second_min_index = np.argsort(dists)[1]
                return arr[second_min_index]
            else:
                return arr[min(enumerate(dists), key=lambda x: x[1])[0]]
        elif len(arr) == 1:
            return arr[0]
        else:
            return None

    def localize_electrodes(self):
        midline_rgb = self._get_electrode_suggestions(use_rgb=True)
        midline_hsv = self._get_electrode_suggestions(use_rgb=False)

        rgb_fpz_suggestion = self._get_fpz(arr=midline_rgb)
        hsv_fpz_suggestion = self._get_fpz(arr=midline_hsv)

        # Check if both are suggestions are None
        if rgb_fpz_suggestion is None and hsv_fpz_suggestion is None:
            # Try to estimate the position of Fpz based on nasion and ray-tracing
            estimated_fpz = self._estimate_fpz()

            # Do not entirely trust the estimated position of FPz
            if estimated_fpz is not None:
                self._fpz = estimated_fpz
            else:
                self.red_flag = True
        # If one of them is None, set FPz to be the other
        elif rgb_fpz_suggestion is None:
            self._fpz = hsv_fpz_suggestion
        elif hsv_fpz_suggestion is None:
            self._fpz = rgb_fpz_suggestion
        else:
            # Check that the points are within 0.5 cm from each-other
            if np.linalg.norm(rgb_fpz_suggestion - hsv_fpz_suggestion) <= 0.005:
                self._fpz = rgb_fpz_suggestion
            else:
                # If not, select the one that are closest to the x-axis from nasion.
                # Calculate the x-axis distance from self._nasion for each point
                x_distance_rgb = np.abs(rgb_fpz_suggestion[0] - self._nasion[0])
                x_distance_hsv = np.abs(hsv_fpz_suggestion[0] - self._nasion[0])

                # Choose the point with the closest x-axis value to self._nasion
                if x_distance_rgb < x_distance_hsv:
                    self._fpz = rgb_fpz_suggestion
                else:
                    self._fpz = hsv_fpz_suggestion

    def plot_fpz(self):
        self._plot_mesh(rotation_matrix=self._rotation_matrix, points=[self._fpz], labels=['FPz'])

    def pick_fpz(self, return_point=False):
        """ This function calls the function to pick a manual point.

        Parameters
        ----------
        Returns
        -------
        np.ndarray
            The point that is nasion, either the picked or the suggested point sent as argument depending on the
            correctness of the initial guess

        """
        suggested_point = self.fpz

        if suggested_point is None:
            picked_point = None
            while picked_point is None:
                picked_point = manual_point_selection(obj_file_path=self._obj_file,
                                                      jpg_file_path=self._jpg_file,
                                                      rotation_matrix=self._rotation_matrix,
                                                      point_transform=self._point_transformation,
                                                      point_to_be_picked="FPz")
            if return_point:
                return picked_point
            else:
                self._fpz = picked_point
        else:
            picked_point = manual_point_selection(obj_file_path=self._obj_file,
                                                  jpg_file_path=self._jpg_file,
                                                  rotation_matrix=self._rotation_matrix,
                                                  point_transform=self._point_transformation,
                                                  point_to_be_picked="FPz",
                                                  suggested_point=suggested_point)
            # If no point is selected, this means that the point was correct, and that is returned
            if picked_point is None:
                if return_point:
                    return suggested_point
                else:
                    self._fpz = suggested_point
            else:
                if return_point:
                    return picked_point
                else:
                    self._fpz = picked_point

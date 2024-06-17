import numpy as np
import trimesh

from pos_3d.classes.meshtransformer import MeshTransformer


class PreAuricular(MeshTransformer):
    def __init__(self, rotation_matrix, orientation_dictionary, ** kwargs):
        super().__init__(rotation_matrix=rotation_matrix, orientation_dictionary=orientation_dictionary, **kwargs)
        self._lpa = None
        self._rpa = None
        self._red_flag = False

    @property
    def rpa(self):
        return self._rpa

    @property
    def lpa(self):
        return self._lpa

    @property
    def red_flag(self):
        return self._red_flag

    def find_lpa_rpa(self, tolerance=0.025):
        """ This function calls the seperate functions to find LPA and RPA.

        1. Get the mesh with lpa/rpa + some noise
        2. Get candidates based on the color white
        3. Evaluate the candidates based on ray tracing, if this detects a pair, use this as rpa and lpa else
         evaluate the candidates based on various configurations -> see function evaluate_candidates
        4. Label the points based on the distance to the right and left side of the mesh, and also enables manual
        picking of points if this is set to True
        5. Try to estimate teh center of the point based on the colors of the neighbors in the surrounding radius
        if manual pick is not set to True


        Parameters
        ----------
        tolerance: float, optional
            defaults to 0.025, value used for keeping parts of the mesh

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The suggestion for the LPA and RPA point in 3D coordinates
        """

        mesh = self._get_sliced_mesh(tolerance=tolerance)
        candidates, colors = self._white_color_search(mesh=mesh, nasion=False, min_distance=0.01)
        eval_candidates = self._evaluate_using_ray_trace(candidates=candidates, mesh=mesh)
        if eval_candidates is None:
            eval_candidates = self._evaluate_candidates(mesh=mesh, candidates=candidates, colors=colors)
        self._lpa, self._rpa = self._label_points(eval_candidates)

        if self._lpa is not None:
            self._lpa = self._find_center_of_point(mesh=mesh, point=self._lpa, radius=0.015)
        if self._rpa is not None:
            self._rpa = self._find_center_of_point(mesh=mesh, point=self._rpa, radius=0.015)
        return self._rpa, self._lpa

    def _label_points(self, candidates):
        """ This function is labeling the electrodes to lpa or rpa. In addition, handles if one of them is None, and
        if the manual pick is set to True.

        Parameters
        ----------
        candidates: list
            Should be two candidate points that are LPA or RPA, It can also be None

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            lpa and rpa

        """
        p1, p2 = candidates

        if p1 is None and p2 is None:
            self._red_flag = True
            self._logger.critical("No candidates for LPA and RPA found")
            return p1, p2
        elif p1 is None:
            # Check distance to the right and left side of the face
            dist_right = np.linalg.norm(p2 - self._right)
            dist_left = np.linalg.norm(p2 - self._left)

            if dist_right < dist_left:
                self._red_flag = True
                self._logger.critical("Only LPA was found")
                return p2, None
            else:
                self._red_flag = True
                self._logger.critical("Only RPA was found")
                return None, p2
        elif p2 is None:
            # Check distance to the right and left side of the face
            dist_right = np.linalg.norm(p1 - self._right)
            dist_left = np.linalg.norm(p1 - self._left)
            if dist_right < dist_left:
                self._red_flag = True
                self._logger.critical("Only LPA was found")
                return p1, None
            else:
                self._red_flag = True
                self._logger.critical("Only RPA was found")
                return None, p1
        else:
            dist_right_p1 = np.linalg.norm(p1 - self._right)
            dist_right_p2 = np.linalg.norm(p2 - self._right)

            if dist_right_p1 < dist_right_p2:
                return p2, p1
            else:
                return p1, p2

    def _get_sliced_mesh(self, tolerance=0.025):
        """ Removes unnecessary parts of the mesh related to finding the lpa and rpa point

        Parameters
        ----------
        tolerance: float, optional
            When slicing the mesh, usually we keep the middle +/- some values of the mesh, that is the tolerance value

        Returns
        -------
        trimesh.Trimesh
            mesh object with less noisy parts
        """
        # Remove the top 1/4 quarter of the head, as the nasion, inion cannot be there
        mesh = self._slice_quarter_from_top()
        # Remove tha back and front of the head, keeping the middle +/- tolerance
        mesh = self._get_center_plus_tolerance(mesh=mesh, axis=1, tolerance=tolerance)
        return mesh

    def _evaluate_candidates(self, mesh, candidates, colors):
        """ This is a rather complex function trying to account for multiple edge cases. WIll be divided into
        multiple functions in the future.

        1. Check length of candidates:
            - if 0: return None,None
            - if 1: Do raytracing, if ray tracing is not None, return candidate, ray-tracing else candidate, None
            - if 2: * If more than 10cm apart and different sign of the z axis, return both
                    * If only distance more than 10 cm, both points on the same side, ray-trace both points, if any
                    of the ray-tracing is None, return the other point and its ray tracing. The first candidate is the
                    most likely to be correct, unless the z-axis is lower and the color difference is low.
            - if N: * Check if there are only one point on one side, if that is the case, return that point and the
                      first candidate.
                    * If all points on the same side, ray-trace the first candidate and return those
                    * Lastly, raytrace from the first candidate and select one of the two nearest points on the other
                    side based on distances and colors

        Parameters
        ----------
        mesh: Trimesh.trimesh
            LPA RPA mesh object
        candidates: list
            candidates point
        colors: list[float]
            closeness to white color

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            points or None

        """
        candidates = list(candidates)
        # If there are no candidates, return None
        if len(candidates) == 0:
            return None, None
        elif len(candidates) == 1:
            # Only one candidate is found, ray trace and attempt to find a white point near the ray-trace point
            ray_traced_candidate = self._ray_tracing(start_point=candidates[0], mesh=mesh, white_point=True)
            # If ray_traced candidate is None, return the candidate and None else return candidate and ray-trace point
            if ray_traced_candidate is None:
                return candidates[0], None
            else:
                return candidates[0], ray_traced_candidate
        elif len(candidates) == 2:
            # Check that the distance is above 10cm and that the sign of the x-value is different
            if np.linalg.norm(candidates[0] - candidates[1]) > 0.10 and np.sign(candidates[0][0]) + np.sign(
                    candidates[1][0]) == 0:
                return candidates[0], candidates[1]
            elif np.linalg.norm(candidates[0] - candidates[1]) > 0.10:
                # They are separated but on the same side, ray_trace both and see results
                ray_traced_candidate0 = self._ray_tracing(start_point=candidates[0], mesh=mesh, white_point=True)
                ray_traced_candidate1 = self._ray_tracing(start_point=candidates[1], mesh=mesh, white_point=True)
                # Check if only one of them is not None, return the pair where the ray tracing were successful
                if ray_traced_candidate0 is None and ray_traced_candidate1 is not None:
                    return candidates[1], ray_traced_candidate1
                elif ray_traced_candidate1 is None and ray_traced_candidate0 is not None:
                    return candidates[0], ray_traced_candidate0
                elif ray_traced_candidate1 is None and ray_traced_candidate0 is None:
                    return candidates[0], None
                else:
                    # Check height...and color difference, if candidate 2 is higher and the color difference is low
                    if candidates[0][2] > candidates[1][2] and abs(colors[0] - colors[1]) < 75:
                        return candidates[1], ray_traced_candidate1
                    else:
                        return candidates[0], ray_traced_candidate0
            else:
                ray_traced_candidate = self._ray_tracing(start_point=candidates[0], mesh=mesh, white_point=True)
                return candidates[0], ray_traced_candidate
        else:
            result = [1 if np.sign(p[0]) > 0 else 0 for p in candidates]

            # Check if one of the point is on the opposite side of the head, this is a clear candidate
            if sum(result) == 1:
                ind = np.argmax(result)
                if ind == 0:
                    return candidates[0], candidates[1]
                elif ind != 0:
                    return candidates[0], candidates[ind]
            elif sum(result) == len(result) - 1:
                ind = np.argmin(result)
                if ind == 0:
                    return candidates[0], candidates[1]
                elif ind != 0:
                    return candidates[0], candidates[ind]
            elif sum(result) == 0 or sum(result) == len(result):
                # All points are one side
                ray_trace_point = self._ray_tracing(start_point=candidates[0], mesh=mesh, white_point=True)
                return candidates[0], ray_trace_point
            else:
                ray_traced_candidate = self._ray_tracing(start_point=candidates[0], mesh=mesh, white_point=True)

                if ray_traced_candidate is None:
                    distances = [np.linalg.norm(np.array(p) - np.array(candidates[0])) for p in candidates]
                    filtered_distances = [(i, d) for i, d in enumerate(distances) if d > 0.1]

                    # Get the minimum value of the remaining distances based on the distances and keep the index
                    min_index, _ = min(filtered_distances, key=lambda x: x[1])
                    return candidates[0], candidates[min_index]
                else:
                    distances = [np.linalg.norm(np.array(p) - np.array(ray_traced_candidate)) for p in candidates]
                    sorted_dists = np.argsort(distances)[:2]
                    distance1, distance2 = distances[sorted_dists[0]], distances[sorted_dists[1]]
                    col1, col2 = colors[sorted_dists[0]], colors[sorted_dists[1]]

                    if (distance1 < distance2 and col1 < col2) or (col1 < col2 and abs(distance1 - distance2) > 0.01):
                        return candidates[0], candidates[sorted_dists[0]]
                    elif col2 < col1:
                        return candidates[0], candidates[sorted_dists[1]]
                    else:
                        return candidates[0], candidates[sorted_dists[0]]

    def _ray_tracing(self, start_point, mesh, white_point=False):
        """ Function for doing ray-tracing

        Parameters
        ----------
        start_point: np.ndarray:
            Where to start the ray tracing from
        mesh: trimesh.Trimesh
            Sliced mesh for the LPA and RPA selection
        white_point: bool, optional
            Defaults to False. If this is True, search in a radius of 5 cm around the ray-traced point for the whitest
            point and return that

        Returns
        -------
        np.ndarray
            Ray trace point
        """

        if np.sign(start_point[0]) > 0:
            direction = np.array([-1, 0, 0])
        else:
            direction = np.array([1, 0, 0])

        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        _, _, locations = intersector.intersects_id(ray_origins=[start_point], ray_directions=[direction],
                                                    return_locations=True)
        # Keep only values that are 5 cm or more away from the original point
        possible_values = [p for p in locations if not np.allclose(start_point, p, atol=0.1)]

        if len(possible_values) == 0:
            # Pass -> No ray-trace point found
            return None
        elif len(possible_values) == 1:
            ray_candidate = possible_values[0]
        else:
            dists = [np.linalg.norm(start_point - p) for p in possible_values]
            # Select the point with the closest distance to the starting point,
            ray_candidate = possible_values[np.argmin(dists)]

        if white_point and ray_candidate is not None:
            ray_candidate = self._find_vertices_with_color(mesh=mesh, point=ray_candidate, radius=0.05)

        return ray_candidate

    def _evaluate_using_ray_trace(self, candidates, mesh):
        """
        Evaluate candidate points using ray tracing to identify closely positioned points.

        Parameters
        ----------
        candidates: list
            Candidate points in some coordinate space.
        mesh : trimesh.Trimesh
            Mesh object that the ray will intersect with.
        Returns
        -------
        tuple[np.ndarray, np.ndarray] or None
            Returns a tuple of two unique candidate points if they are close after ray tracing.
            Returns None if all candidates are on the same side or if no close pairs are found.

        """
        candidates = list(candidates)
        # Classify points based on the sign of their x-coordinates: 1 if x > 0, else 0
        result = [1 if np.sign(p[0]) > 0 else 0 for p in candidates]
        # Check if all candidates are on one side of the mesh
        if sum(result) == 0 or sum(result) == len(candidates):
            # All points are on the same side
            return None
        else:
            # Perform ray tracing for each candidate point
            ray_list = [self._ray_tracing(start_point=p, mesh=mesh) for p in candidates]
            # Define a distance threshold (in the same units as the candidate points)
            threshold_distance = 0.02  # 2 cm away

            # Initialize a set to store unique pairs of indices
            unique_pairs = set()

            # Loop through ray-traced points
            for i, ray_point in enumerate(ray_list):
                if ray_point is not None:
                    # Loop through candidates to check if any are close to the ray_point
                    for j, cand in enumerate(candidates):
                        # Calculate distance between ray_point and candidate
                        distance = np.linalg.norm(np.array(ray_point) - np.array(cand))
                        if distance < threshold_distance:
                            # Add the pair to the unique set (frozenset ensures [0,1] == [1,0])
                            unique_pairs.add(frozenset([i, j]))

            # Convert the unique pairs set to a list of lists
            unique_pairs = [list(pair) for pair in unique_pairs]

            # Return either None or the first unique pair of candidates found
            if len(unique_pairs) == 0:
                return None
            else:
                return candidates[unique_pairs[0][0]], candidates[unique_pairs[0][1]]

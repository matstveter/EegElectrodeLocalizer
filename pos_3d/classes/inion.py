import numpy as np
import trimesh

from pos_3d.classes.meshtransformer import MeshTransformer


class Inion(MeshTransformer):
    def __init__(self, rotation_matrix, orientation_dictionary, nasion, lpa, rpa, manual_pick=False, **kwargs):
        super().__init__(rotation_matrix=rotation_matrix, orientation_dictionary=orientation_dictionary, **kwargs)
        self._red_flag = False
        self._origo = None
        self._inion = None
        self._nasion = nasion
        self._rpa = rpa
        self._lpa = lpa
        self._manual_pick = manual_pick

    @property
    def inion(self):
        return self._inion

    @property
    def red_flag(self):
        return self._red_flag

    @property
    def origo(self):
        return self._origo

    def estimate_inion(self):
        """ Function to estimate the inion position calling the required functions inside the class, and returns
        the estimated inion position.
        """
        self._origo, direction = self._find_direction_and_origo()
        if self._origo is None:
            self._red_flag = True
            return None
        suggested_inion = self._ray_tracing(direction)

        if suggested_inion is None:
            self._logger.critical("Can't find inion points from ray-tracing or manually")
            self._red_flag = True

        self._inion = suggested_inion
        return self._inion

    def _ray_tracing(self, direction):
        """ Crate a ray-trace object and use the mesh and the direction sent as input to find potential candidates
        for inion. If there are non-potential candidates, try to estimate it another way using the function
        self._manual_estimation_of_inion(), or if there are potential candidates return the nearest one.

        Parameters
        ----------
        direction: np.ndarray
            3-dimensional direction vector

        Returns
        -------
        np.ndarray
            Point which is the inion suggestion

        """
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self._mesh.copy())
        _, _, locations = intersector.intersects_id(ray_origins=[self._nasion],
                                                    ray_directions=[np.array(direction)],
                                                    return_locations=True)
        possible_values = [p for p in locations if not np.allclose(self._nasion, p, atol=0.1)]

        if len(possible_values) == 0:
            return self._manual_estimation_of_inion()
        else:
            # Return the point nearest to the starting
            return possible_values[0]

    def _find_direction_and_origo(self):
        """ Find the closest point to the nasion on the line connecting LPA and RPA,
        and determine the direction from the nasion to this point.

        This function is useful for establishing an origin and direction for subsequent computations,
        such as ray-tracing algorithms to find the inion.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            The closest point to the nasion on the line (suggested_origo) and the direction vector from
            the nasion to this point (direction).
        """

        def closest_point_on_line(A, B, P):
            """
            Find the closest point on a line defined by points A and B to a point P.

            Parameters
            ----------
            A : np.ndarray
                One end-point of the line.
            B : np.ndarray
                The other end-point of the line.
            P : np.ndarray
                The point to project onto the line.

            Returns
            -------
            np.ndarray
                The closest point on the line to point P.
            """
            AB = B - A
            AP = P - A
            t = np.dot(AP, AB) / np.dot(AB, AB)
            return A + t * AB

        if self._lpa is not None and self._rpa is not None and self._nasion is not None:
            suggested_origo = closest_point_on_line(A=self._lpa, B=self._rpa, P=self._nasion)
        else:
            return None, None
        direction = suggested_origo - self._nasion
        return suggested_origo, direction

    def _manual_estimation_of_inion(self):
        """ Creates a plane, and searches for the point furthest away from the nasion point on the plane, within a
        tolerance value in the x-axis from nasion.

        Returns
        -------
        np.ndarray
            inion estimated point in 3D space
        """
        plane_normal = np.cross(self._lpa - self._nasion, self._rpa - self._nasion)
        plane_point = self._nasion  # any point on the plane
        # Search the mesh vertices to find the point most distant from the nasion
        # but still lying on the plane.
        max_distance = 0
        tolerance = 1e-3
        inion_estimated = None
        for vertex in self._mesh.vertices:
            if abs(np.dot(plane_normal, vertex - plane_point)) < tolerance:
                # Check if x-coordinate is almost equal to nasion x-coordinate
                if np.abs(vertex[0] - self._nasion[0]) < tolerance:
                    distance = np.linalg.norm(vertex - self._nasion)
                    if distance > max_distance:
                        max_distance = distance
                        inion_estimated = vertex
        return inion_estimated

    def inion_correction(self):
        raise NotImplementedError("Future shift of lpa, rpa and nasion to better create a coordinate system...")

    def plot_solution(self):
        self._plot_mesh(rotation_matrix=self._rotation_matrix,
                        points=[self._nasion, self._inion, self._lpa, self._rpa],
                        labels=['Nasion', 'Inion', 'LPA', 'RPA'])

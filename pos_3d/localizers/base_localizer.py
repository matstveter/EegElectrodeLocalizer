import abc
import trimesh
import numpy as np

from pos_3d.classes.meshtransformer import MeshTransformer


class BaseLocalizer(MeshTransformer):
    def __init__(self, rotation_matrix, orientation_dictionary, nasion, inion, lpa, rpa, origo, **kwargs):
        super().__init__(rotation_matrix=rotation_matrix, orientation_dictionary=orientation_dictionary, **kwargs)
        self._red_flag = False
        self._origo = origo,
        self._nasion, self._inion, self._lpa, self._rpa = nasion, inion, lpa, rpa

    def _get_mesh(self, update_faces=False, plot_mesh=False):
        """
        Get the portion of the mesh that lies on one side of the plane defined by
        the points nasion, lpa, and rpa.

        Parameters
        ----------
        update_faces : bool, optional
            Whether to update the faces of the mesh based on the vertex mask.
            Default is False.
        plot_mesh: bool, optional
            Plot the sliced mesh
            Default is False

        Returns
        -------
        trimesh.Trimesh
            The modified mesh.

        Notes
        -----
        The plane is defined by the equation ax + by + cz + D = 0, where (a, b, c) is
        the normal to the plane, and D is a constant. Points on one side of the plane
        satisfy the inequality ax + by + cz + D > 0.
        """
        # Compute the normal to the plane defined by nasion, lpa, and rpa
        plane_normal = np.cross((np.array(self._lpa) - np.array(self._nasion)),
                                (np.array(self._rpa) - np.array(self._nasion)))

        # Normalize the plane normal
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Compute the constant term D in the plane equation
        D = -np.dot(plane_normal, self._nasion)

        # Shift to capture a bit more of the electrodes
        D = D + 0.015

        # Copy the head mesh
        head_mesh = self._head_mesh.copy()

        # Create a mask identifying which vertices lie on one side of the plane
        vertex_mask = np.dot(head_mesh.vertices, plane_normal) + D > 0

        # Optionally update the faces of the mesh
        if update_faces:
            # Create a mask for faces where all vertices satisfy the inequality
            face_mask = np.all(vertex_mask[head_mesh.faces], axis=1)
            # Update the mesh faces based on the face mask
            head_mesh.update_faces(face_mask)

        # Update the mesh vertices based on the vertex mask
        head_mesh.update_vertices(vertex_mask)

        if plot_mesh:
            head_mesh.show()
        return head_mesh

    @abc.abstractmethod
    def localize_electrodes(self):
        pass


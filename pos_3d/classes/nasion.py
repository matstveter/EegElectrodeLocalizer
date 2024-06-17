import numpy as np

from pos_3d.classes.meshtransformer import MeshTransformer


class Nasion(MeshTransformer):
    def __init__(self, rotation_matrix, orientation_dictionary, **kwargs):
        super().__init__(rotation_matrix=rotation_matrix, orientation_dictionary=orientation_dictionary, **kwargs)
        self._red_flag = False
        self._nasion = None

    @property
    def nasion(self):
        return self._nasion

    @property
    def red_flag(self):
        return self._red_flag

    def find_nasion(self, tolerance=0.02):
        """ This function calls the seperate functions to find nasion.

        1. Get the mesh with nasion + some noise
        2. Get candidates based on the color white
        3. Evaluate the candidates based on whiteness, and position
        4. Try to estimate teh center of the point based on the colors of the neighbors in the surrounding radius
        5. If manual pick is set to True, allow for manual verification of nasion point

        Parameters
        ----------
        tolerance: float, optional
            defaults to 0.02, value used for keeping parts of the mesh

        Returns
        -------
        np.ndarray
            The suggestion for the nasion point in 3D coordinates
        """
        mesh = self._get_sliced_mesh(tolerance=tolerance)
        candidates, colors = self._white_color_search(mesh=mesh, nasion=True, min_distance=0.02,
                                                      max_candidates=3)
        temp_nasion = self._evaluate_candidates(candidates=candidates, colors=colors)

        # If there are no candidates for nasion, attempt to soften the criteria and see if that helps, 
        # raise warning if not, or at a later stage, flag this subject as not correct
        if temp_nasion is None:
            self._logger.critical("Did not find any candidates for nasion, trying to run again, with softer criteria")
            candidates, colors = self._white_color_search(mesh=mesh, nasion=True, min_distance=0.01,
                                                          max_candidates=3, white_cutoff_distance=250)
            if len(candidates) == 0:
                self._logger.critical("Still no nasion found, raising error")
                self._red_flag = True
            else:
                temp_nasion = candidates[0]

        # This function tries to look for similar neighbors in the surrounding area, and tries to find the center
        # of the actual nasion point
        temp_nasion = self._find_center_of_point(mesh=mesh, point=temp_nasion, radius=0.015)

        # If we want to manually be able to pick the point, call function for that, or set it to be the temp_nasion
        self._nasion = temp_nasion

        return self._nasion

    def _get_sliced_mesh(self, tolerance=0.02):
        """ Removes unnecessary parts of the mesh related to finding the nasion point

        Parameters
        ----------
        tolerance: float, optional
            When slicing the mesh, usually we keep the middle +/- some values of the mesh, that is the tolerance value

        Returns
        -------
        trimesh.Trimesh
            mesh object with less noisy parts

        """
        # Remove the top 1/4 of the head, as the nasion cannot be there
        mesh = self._slice_quarter_from_top()
        # Remove everything except the face, as the nasion cannot be anywhere else
        mesh = self._get_face_slice(mesh=mesh, tolerance=tolerance)
        # Remove the sides of the face, except the middle +/- tolerance, as the nasion cannot be on the sides
        mesh = self._get_center_plus_tolerance(mesh=mesh, axis=0, tolerance=tolerance)
        return mesh

    @staticmethod
    def _evaluate_candidates(candidates, colors):
        """ This function attempts to choose the correct point from a set of candidates based on color, and
        relational position between them

        Parameters
        ----------
        candidates: list
            candidates in 3D coordinates
        colors: list
            Color differences from white from the candidates

        Returns
        -------
        np.ndarray
            The suggested nasion point
        """
        candidates = list(candidates)
        if len(candidates) == 0:
            return None
        elif len(candidates) == 1:
            return candidates[0]
        else:
            if len(candidates) > 2:
                # Find an approach for sub-selecting if all colors is below 100 for instance
                # If all points are below 100 away from white and there are three points,
                # choose the top two for further analysis, as this is probably 2x face mask and nasion
                if all(c < 100 for c in colors):
                    zipped = zip(candidates, colors)
                    sorted_list = sorted(zipped, key=lambda x: x[0][2], reverse=True)
                    candidates, colors = zip(*sorted_list)

                # Get the 2 most likely candidates for further analysis
                candidates = candidates[:2]
                colors = colors[:2]

            candidates = np.array(candidates)
            colors = np.array(colors)
            color_diff = abs(colors[0] - colors[1])

            # If both i 100 away from white, suspicion is poor lighting and nasion-electrode pair
            if colors[0] > 100 and colors[1] > 100:
                if color_diff < 65:
                    return candidates[np.argmax(candidates[:, 2])]
                else:
                    return candidates[np.argmin(candidates[:, 2])]
            # Either face mask and nasion, or nasion and gloves
            elif color_diff < 65 or colors[1] > 175:
                return candidates[np.argmax(candidates[:, 2])]
            else:
                return candidates[np.argmin(candidates[:, 2])]

import csv
import os
import numpy as np

from pos_3d.classes.inion import Inion
from pos_3d.classes.meshtransformer import MeshTransformer
from pos_3d.utils.automatic_verification import get_landmark_from_csv
from pos_3d.utils.mesh_helper_functions import manual_point_selection


class LandmarkVerifier(MeshTransformer):

    def __init__(self, rotation_matrix, orientation_dictionary, nasion, inion, lpa, rpa, origo, target_folder,
                 true_positions=None, **kwargs):
        super().__init__(rotation_matrix=rotation_matrix, orientation_dictionary=orientation_dictionary, **kwargs)
        self._true_positions = true_positions
        self._nasion = nasion
        self._inion = inion
        self._lpa = lpa
        self._rpa = rpa
        self._origo = origo
        self._kwargs = kwargs

        self._target_folder = target_folder

    @property
    def nasion(self):
        return self._nasion

    @property
    def inion(self):
        return self._inion

    @property
    def lpa(self):
        return self._lpa

    @property
    def rpa(self):
        return self._rpa

    @property
    def origo(self):
        return self._origo

    def _pick_single_electrode(self, name, suggested_point):
        """ This function calls the function to pick a manual point.

        Parameters
        ----------
        suggested_point: [None, np.ndarray], optional
            None is default, but if it is not None it is a suggestion the nasion point

        Returns
        -------
        np.ndarray
            The point that is nasion, either the picked or the suggested point sent as argument depending on the
            correctness of the initial guess

        """
        if suggested_point is None:
            picked_point = None
            while picked_point is None:
                picked_point = manual_point_selection(obj_file_path=self._obj_file,
                                                      jpg_file_path=self._jpg_file,
                                                      rotation_matrix=self._rotation_matrix,
                                                      point_transform=self._point_transformation,
                                                      point_to_be_picked=name)
            return picked_point
        else:
            picked_point = manual_point_selection(obj_file_path=self._obj_file,
                                                  jpg_file_path=self._jpg_file,
                                                  rotation_matrix=self._rotation_matrix,
                                                  point_transform=self._point_transformation,
                                                  point_to_be_picked=name,
                                                  suggested_point=suggested_point)
            # If no point is selected, this means that the point was correct, and that is returned
            if picked_point is None:
                return suggested_point
            else:
                return picked_point

    def write_to_csv(self, estimated=True):
        """
        Write landmark coordinates to a CSV file.

        This method exports the coordinates of specific anatomical landmarks (Nasion, RPA, LPA)
        into a CSV file. The file is named either 'estimated_landmark.csv' or 'validated_landmark.csv',
        depending on the value of the 'estimated' parameter.

        Parameters
        ----------
        estimated : bool, optional
            If True, the method writes the estimated landmark positions to the CSV file.
            If False, it writes the validated positions. Default is True.

        Notes
        -----
        The method relies on instance attributes `_nasion`, `_rpa`, `_lpa` for the landmark
        coordinates, and `_target_folder` for the destination folder of the CSV file.
        The CSV file contains a header row with the fields 'Landmark', 'x', 'y', 'z',
        followed by rows for each of the landmarks.
        """
        if estimated:
            string = "estimated"
        else:
            string = "validated"

        if self._nasion is None:
            self._nasion = [0, 0, 0]
            self._logger.error("Nasion is None, set to [0, 0, 0]")
        if self._lpa is None:
            self._lpa = [0, 0, 0]
            self._logger.error("LPA is None, set to [0, 0, 0]")
        if self._rpa is None:
            self._rpa = [0, 0, 0]
            self._logger.error("RPA is None, set to [0, 0, 0]")

        fields = ['Landmark', 'x', 'y', 'z']
        data = [
            ['Nasion', f"{self._nasion[0]:.6f}", f"{self._nasion[1]:.6f}", f"{self._nasion[2]:.6f}"],
            ['RPA', f"{self._rpa[0]:.6f}", f"{self._rpa[1]:.6f}", f"{self._rpa[2]:.6f}"],
            ['LPA', f"{self._lpa[0]:.6f}", f"{self._lpa[1]:.6f}", f"{self._lpa[2]:.6f}"],
        ]

        with open(os.path.join(self._target_folder, f"{string}_landmark.csv"), mode="w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write the header row
            csvwriter.writerow(fields)
            # Write each row of data
            for row in data:
                csvwriter.writerow(row)

    def manual_pick(self):
        """ Function that is called when the manual_pick flag is True, so that the user can verify each of the positions
        of the landmarks and make adjustments. After the three positions are picked, inion and origo is estimated.

        Returns
        -------

        """

        self._nasion = self._pick_single_electrode(name="Nasion", suggested_point=self._nasion)
        self._lpa = self._pick_single_electrode(name="LPA", suggested_point=self._lpa)
        self._rpa = self._pick_single_electrode(name="RPA", suggested_point=self._rpa)

        # Estimate a new inion and origo
        inion = Inion(orientation_dictionary=self._orientation_dictionary,
                      rotation_matrix=self._rotation_matrix, nasion=self._nasion,
                      rpa=self._rpa, lpa=self._lpa, **self._kwargs)
        inion.estimate_inion()
        self._inion = inion.inion
        self._origo = inion.origo

    def plot_solution(self):
        self._plot_mesh(rotation_matrix=self._rotation_matrix, points=[self._nasion, self._lpa, self._rpa, self._inion],
                        labels=['Nasion', 'LPA', 'RPA', 'Inion'])

    def get_found_landmarks(self, path):
        positions = get_landmark_from_csv(path=path)
        self._nasion = np.array(positions['Nasion'])
        self._rpa = np.array(positions['RPA'])
        self._lpa = np.array(positions['LPA'])

        # Estimate a new inion and origo
        inion = Inion(orientation_dictionary=self._orientation_dictionary,
                      rotation_matrix=self._rotation_matrix, nasion=self._nasion,
                      rpa=self._rpa, lpa=self._lpa, **self._kwargs)
        inion.estimate_inion()
        self._inion = inion.inion
        self._origo = inion.origo

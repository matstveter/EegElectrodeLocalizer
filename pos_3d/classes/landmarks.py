import numpy as np

from pos_3d.classes.inion import Inion
from pos_3d.classes.nasion import Nasion
from pos_3d.classes.pre_auricular import PreAuricular


class LandmarksClass:
    def __init__(self, orientation_dictionary, rotation_matrix, **kwargs):
        self._kwargs = kwargs
        self._nasion = None
        self._inion = None
        self._rpa = None
        self._lpa = None
        self._origo = None

        self._orientation_dictionary = orientation_dictionary
        self._rotation_matrix = rotation_matrix

        self._nasion_object, self._preauricular_object, self._inion_object = None, None, None

    @property
    def landmarks(self):
        return self._nasion, self._inion, self._rpa, self._lpa

    @property
    def nasion(self):
        return self._nasion

    @property
    def inion(self):
        return self._inion

    @property
    def rpa(self):
        return self._rpa

    @property
    def lpa(self):
        return self._lpa

    @property
    def origo(self):
        return self._inion_object.origo

    @property
    def red_flag(self):
        return self._nasion_object.red_flag or self._inion_object.red_flag or self._preauricular_object.red_flag

    def get_landmarks(self):
        self._nasion = self._find_nasion()
        self._rpa, self._lpa = self._find_lpa_rpa()
        self._inion = self._inion_estimator()

    def _find_nasion(self):
        """ Creates a nasion object, and returns the nasion

        Returns
        -------
        np.ndarray
            (1, 3) 3D coordinates for the nasion point
        """
        n = Nasion(orientation_dictionary=self._orientation_dictionary,
                   rotation_matrix=self._rotation_matrix, **self._kwargs)
        n.find_nasion()
        self._nasion_object = n
        return n.nasion

    def _find_lpa_rpa(self):
        """ Creates a PreAuricular class and returns the found RPA and LPA

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            RPA and LPA as 2 3D points

        """
        pa = PreAuricular(orientation_dictionary=self._orientation_dictionary,
                          rotation_matrix=self._rotation_matrix, **self._kwargs)
        pa.find_lpa_rpa()
        self._preauricular_object = pa
        return pa.rpa, pa.lpa

    def _inion_estimator(self):
        """ Creates an inion object, and returns the inion point

        Returns
        -------
        np.ndarray
            (1, 3) 3D coordinates for the inion point
        """
        inion = Inion(orientation_dictionary=self._orientation_dictionary,
                      rotation_matrix=self._rotation_matrix, nasion=self._nasion,
                      rpa=self._rpa, lpa=self._lpa, **self._kwargs)
        inion.estimate_inion()
        self._inion_object = inion
        self._origo = inion.origo
        return inion.inion

    def plot_solution(self):
        self._inion_object.plot_solution()

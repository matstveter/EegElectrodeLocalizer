from math import atan2, sqrt

import numpy as np


def calculate_polar_coordinates(vec1, vec2):
    """
    Calculate the polar coordinates of vec2 relative to vec1 in 3D space.

    Parameters:
    vec1 (array-like): The x, y, z coordinates of the reference point.
    vec2 (array-like): The x, y, z coordinates of the point to which we are calculating the polar coordinates.

    Returns:
    tuple: A tuple containing the radial distance (r), azimuthal angle (theta), and polar angle (phi).
    """
    # Calculate the differences in each coordinate
    dx = vec2[0] - vec1[0]
    dy = vec2[1] - vec1[1]
    dz = vec2[2] - vec1[2]

    # Calculate the radial distance (Euclidean distance) from vec1 to vec2
    r = sqrt(dx*dx + dy*dy + dz*dz)

    # Calculate the azimuthal angle (theta) in the xy-plane from the positive x-axis
    # It ranges from -pi to pi
    theta = atan2(dy, dx)

    # Calculate the polar angle (phi) from the positive z-axis
    # It ranges from 0 to pi
    phi = atan2(sqrt(dx*dx + dy*dy), dz)

    return r, theta, phi


def create_electrode_dict(temp_dict):
    """
    Create a dictionary of ElectrodePoint objects and populate their neighbor attributes.

    This function takes a dictionary containing electrode names and their 3D coordinates,
    creates an ElectrodePoint object for each, and calculates the Euclidean distance and
    polar coordinates between each pair of electrodes. These metrics are stored in each
    ElectrodePoint object's '_neighbors' attribute.

    Parameters
    ----------
    temp_dict : dict
        A dictionary containing electrode names as keys and their 3D coordinates as values.
        For example, {'E1': [x1, y1, z1], 'E2': [x2, y2, z2]}.

    Returns
    -------
    dict
        A dictionary where the keys are electrode names and the values are corresponding
        ElectrodePoint objects with populated '_neighbors' attributes.
    """

    # Initialize an empty dictionary to store ElectrodePoint objects.
    electrodes = {}

    # Populate the dictionary with ElectrodePoint objects using data from temp_dict.
    for key, val in temp_dict.items():
        electrodes[key] = ElectrodePoint(
            name=key,
            temp_x=val[0],
            temp_y=val[1],
            temp_z=val[2]
        )

    # Loop through each pair of electrodes to calculate distance and polar coordinates.
    for key_1, el_1 in electrodes.items():
        for key_2, el_2 in electrodes.items():
            if key_1 == key_2:
                continue

            # Calculate the Euclidean distance between electrode pairs.
            distance = np.linalg.norm(el_1.template_coordinates - el_2.template_coordinates)

            # Calculate the polar coordinates for the electrode pairs.
            r, theta, phi = calculate_polar_coordinates(
                vec1=el_1.template_coordinates,
                vec2=el_2.template_coordinates
            )

            # Populate the '_neighbors' attribute of each ElectrodePoint object.
            el_1.add_neighbors(
                name=key_2,
                pos=el_2.template_coordinates,
                distance=distance,
                r=r,
                theta=theta,
                phi=phi
            )
    return electrodes


class ElectrodePoint:
    def __init__(self, name, temp_x, temp_y, temp_z):
        self._name = name
        self._template_coordinates = np.array([temp_x, temp_y, temp_z])
        self._coordinates = np.array([0, 0, 0])
        self._color = None
        self._neighbors = {}

    @property
    def name(self):
        return self._name

    @property
    def template_coordinates(self):
        return self._template_coordinates

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):
        self._coordinates = value

    def add_neighbors(self, name, pos, distance, r, theta, phi):
        self._neighbors[name] = {
            'label': name,
            'pos': pos,
            'dist': distance,
            'r': r,
            'theta': theta,
            'phi': phi
        }

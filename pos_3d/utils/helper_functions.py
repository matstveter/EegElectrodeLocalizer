import os
from datetime import datetime
import numpy as np
import logging
import open3d as o3d

from pycpd import RigidRegistration


def get_mesh_files(dataset_path, subject_folder, texture_file_format="jpg", mesh_format="obj"):
    """
    Get the paths for image and mesh files, checking if they exist and allowing for different file formats.

    Args:
    dataset_path (str): Path to the dataset.
    subject_folder (str): Folder for the specific subject.
    img_format (str, optional): Image file format (default is "jpg").
    mesh_format (str, optional): Mesh file format (default is "obj").

    Returns:
    tuple: Paths to the image and mesh files if they exist.

    Raises:
    FileNotFoundError: If the image or mesh file does not exist.
    """
    texture_file = os.path.join(dataset_path, subject_folder, f"Model.{texture_file_format}")
    mesh_file = os.path.join(dataset_path, subject_folder, f"Model.{mesh_format}")

    if not os.path.isfile(texture_file):
        raise FileNotFoundError(f"The image file {texture_file} does not exist.")
    if not os.path.isfile(mesh_file):
        raise FileNotFoundError(f"The mesh file {mesh_file} does not exist.")

    return texture_file, mesh_file


def create_subject_folder(output_path, subject_id):
    """

    Parameters
    ----------
    output_path: str
        Path to the output folder, where each subject will have a sub folder containing their specific information
    subject_id: str
        ID of the subject, typically 1-000-1-A

    Returns
    -------
    str
        path to the subject specific result folder
    logging.Logger
        logging object for this subject
    """
    # Create a folder with the subject_id as the name
    folder_path = os.path.join(output_path, subject_id)
    os.makedirs(folder_path, exist_ok=True)

    # Create a logging file
    logging.basicConfig(filename=os.path.join(folder_path, "log.log"), level=logging.INFO, filemode="w",
                        format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    logging.info("Created folder, geo file, electrode file and logging setup!")
    return folder_path, logging


def create_run_folder(path=""):
    """
    Create a timestamped folder within a specified directory.

    Parameters:
    -----------
    path : str
        The base directory in which the timestamped folder will be created.

    Returns:
    --------
    str
        The path to the newly created timestamped folder.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H_%M_%S')

    if path == "":
        run_path = os.path.join(os.getcwd(), "results/", timestamp)
    else:
        run_path = os.path.join(path, timestamp)

    os.mkdir(run_path)
    return run_path


def get_distance_between_points(p1, p2):
    """ Method to calculate distance between two points

    Parameters
    ----------
    p1: np.ndarray
        point in 3d space
    p2: np.ndarray
        point in 3d space
    Returns
    -------
        distance between the points
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def read_elc_file(file_path, scale):
    template_dict = {}
    labels = []
    pos_list = []

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                label, coord_str = line.strip().split('\t:\t')
                x, y, z = map(float, coord_str.split('\t'))
                pos = np.array([x, y, z]) / scale
                template_dict[label] = pos
                labels.append(label)
                pos_list.append(pos)

        # Estimate nasion, using the 1010 system, the distance between FPz should be the same as the distance between
        # AFz and FPz
        distance_from_fpz_to_afz = np.linalg.norm(template_dict['AFz'] - template_dict['FPz'])
        x, y, z = template_dict['FPz']
        # Set the x-value a bit higher than Fpz, to make labeling easier later
        template_dict['Nz'] = np.array([(x + 1e-6), y, (z - distance_from_fpz_to_afz)])
        return template_dict, np.array(labels), np.array(pos_list)
    else:
        raise FileNotFoundError(f".elc file not found at this location: {file_path}")


def compare_hist(hist1, hist2):
    intersection = np.minimum(hist1, hist2).sum()
    union = np.maximum(hist1, hist2).sum()
    return intersection / union


def find_electrode_neighbors(electrode_dict, radius=0.04, key=None):
    neighbours_dict = {}

    for e1_name, e1_pos in electrode_dict.items():
        neigh = {}
        if key is not None:
            if key != e1_name:
                continue
        for e2_name, e2_pos in electrode_dict.items():
            if e1_name == e2_name:
                # Skip current electrode
                continue

            temp_dist = np.linalg.norm(e1_pos - e2_pos)
            if temp_dist < radius or (e1_name == "Iz" and e2_name == "POz"):
                unit_vector = (e1_pos - e2_pos) / temp_dist if temp_dist != 0 else np.array([0, 0, 0])
                neigh[e2_name] = {'dist': temp_dist, 'unit_direction': unit_vector}
        neighbours_dict[e1_name] = neigh

    if key is not None:
        return neighbours_dict[key]
    else:
        return neighbours_dict


def get_cutoff_axis(centroid, point):
    """ Find along which axis the point changes the most. Used for creating masks for example from the centroid,
    towards the face of the subject

    Parameters
    ----------
    centroid
    point

    Returns
    -------

    """
    return np.argmax((np.abs(centroid - point)))


def normalize_vector(v):
    return v / np.linalg.norm(v)


def get_rotation_matrix(axis, angle):
    """
    Create a rotation matrix for rotating around the principal axes.

    Parameters:
    - axis: int, the axis to rotate around (0 for x-axis, 1 for y-axis, 2 for z-axis)
    - angle: float, the angle in radians to rotate by

    Returns:
    - A 3x3 numpy.ndarray representing the rotation matrix.
    """
    # Initialize identity matrix
    rotation_matrix = np.eye(3)

    # Calculate cos and sin of the angle
    c, s = np.cos(angle), np.sin(angle)

    # Update the rotation matrix based on the axis
    if axis == 0:
        # Rotation around x-axis
        rotation_matrix[1, 1] = c
        rotation_matrix[1, 2] = -s
        rotation_matrix[2, 1] = s
        rotation_matrix[2, 2] = c
    elif axis == 1:
        # Rotation around y-axis
        rotation_matrix[0, 0] = c
        rotation_matrix[0, 2] = s
        rotation_matrix[2, 0] = -s
        rotation_matrix[2, 2] = c
    elif axis == 2:
        # Rotation around z-axis
        rotation_matrix[0, 0] = c
        rotation_matrix[0, 1] = -s
        rotation_matrix[1, 0] = s
        rotation_matrix[1, 1] = c
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

    return rotation_matrix


def get_sub_dict_from_single_key(dictionary, key):
    sub_dict = {k: v for k, v in dictionary.items() if key in k}
    return sub_dict


def get_two_key_midline_sub_dict(dictionary):
    selected_values = {}  # Initialize an empty dictionary to store selected key-value pair
    for key, value in dictionary.items():
        if len(key) == 2 and (key[0] == 'C' or key[0] == 'T') and key[1].isdigit() or key == "Cz":
            selected_values[key] = value
    return selected_values


def cpd(aligned_template_dict, suggested_electrode_positions):
    """
    Perform Coherent Point Drift (CPD) registration between template electrode positions and suggested electrode
    positions.

    This method applies a Rigid CPD algorithm to align the template electrode positions to the suggested positions.
    The method updates the template positions based on the CPD transformation and returns a
    dictionary with the new positions.

    Parameters
    ----------
    aligned_template_dict : dict
        A dictionary where keys are electrode labels and values are their corresponding 3D positions in
        the template.
        Example format: {'Fz': [0.0, 1.0, 2.0], 'Cz': [1.0, 2.0, 3.0]}.

    suggested_electrode_positions : list of lists
        A list of 3D positions for the suggested electrode locations.
        Example format: [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]].

    Returns
    -------
    dict
        A dictionary with updated electrode positions post CPD registration. Format is similar to
        `aligned_template_dict`.

    Notes
    -----
    This method uses a rigid CPD algorithm, meaning the transformation includes rotation and translation but not
    scaling or shearing.

    The method requires the `RigidRegistration` class from an external CPD library.

    """
    # Convert data to numpy arrays
    template_pos = np.array([*aligned_template_dict.values()])
    suggested_pos = np.array(suggested_electrode_positions)

    # Apply CPD
    reg = RigidRegistration(**{'X': template_pos, 'Y': suggested_pos})
    reg.register()

    # Get the transformed points
    transformed_template = reg.transform_point_cloud(template_pos)

    # Process the results similar to your ICP-based function
    new_dict = {label: point for label, point in zip(aligned_template_dict.keys(), transformed_template)}

    return new_dict


def icp(aligned_template_dict, suggested_electrode_positions, max_correspondence_dist=0.015):
    """
    Perform Iterative Closest Point (ICP) registration to align template electrode positions with suggested
    electrode positions.

    This method uses the ICP algorithm to find an optimal alignment (rotation and translation) of the template
    electrode positions to the suggested positions. It then updates the electrode labels based on the closest
    points in the aligned positions.

    Parameters
    ----------
    aligned_template_dict : dict
        A dictionary where keys are electrode labels and values are their corresponding 3D positions
        in the template.
        Example format: {'Fz': [0.0, 1.0, 2.0], 'Cz': [1.0, 2.0, 3.0]}.

    suggested_electrode_positions : list of lists
        A list of 3D positions for the suggested electrode locations.
        Example format: [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]].

    max_correspondence_dist : float, optional
        The maximum distance threshold for matching points between the template and suggested positions.
        Default value is 0.015.

    Returns
    -------
    tuple of (dict, dict)
        A tuple containing two dictionaries:
        1. A dictionary mapping electrode labels to the closest electrode positions in the suggested list,
        if within the max_correspondence_dist.
        2. A dictionary mapping original template labels to their new positions post ICP registration.

    Notes
    -----
    The method uses Open3D's implementation of the ICP algorithm.


    The returned dictionaries are useful for understanding the alignment and for further processing the
    electrode data.
    """
    potential_el_pcd = o3d.geometry.PointCloud()
    potential_el_pcd.points = o3d.utility.Vector3dVector(np.array(suggested_electrode_positions))

    template_pos = np.array([*aligned_template_dict.values()])
    template_labels = list(aligned_template_dict.keys())
    template_pcd = o3d.geometry.PointCloud()
    template_pcd.points = o3d.utility.Vector3dVector(template_pos)

    icp_result = o3d.pipelines.registration.registration_icp(
        template_pcd,
        potential_el_pcd,
        max_correspondence_distance=max_correspondence_dist,
        init=np.identity(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )

    # The transformation that aligns template_pcd to suggested_electrode_positions_pcd
    transformation = icp_result.transformation

    # Transform the template_pcd to the suggested_electrode_positions_pcd coordinate frame
    aligned_template_pcd = template_pcd.transform(transformation)

    new_dict = {key: np.asarray(aligned_template_pcd.points[i]) for i, key in
                enumerate(aligned_template_dict.keys())}

    electrode_labels = {}
    for i, point in enumerate(aligned_template_pcd.points):
        # Convert to numpy array for ease of manipulation
        query_point = np.asarray(point)

        # Compute the distances from this point to all points in potential_el_pcd
        distances = np.linalg.norm(np.asarray(potential_el_pcd.points) - query_point, axis=1)

        # Find the index of the closest point in potential_el_pcd
        closest_point_idx = np.argmin(distances)

        # If the closest point is within the acceptable range (max_correspondence_distance)
        if distances[closest_point_idx] < max_correspondence_dist:
            # Assign the label from the template to the closest electrode point
            electrode_labels[template_labels[i]] = suggested_electrode_positions[closest_point_idx]

    # Now, electrode_labels dictionary contains the electrode positions with their corresponding labels
    return electrode_labels, new_dict


def update_dictionary(old_dict, new_dict):
    for key, val in new_dict.items():
        old_dict[key] = val
    return old_dict


def calculate_pairwise_distances(electrode_positions, neighbor_dict):
    distances = {}
    for elec, neighbors in neighbor_dict.items():
        elec_pos = np.array(electrode_positions[elec])
        for neighbor in neighbors:
            neighbor_pos = np.array(electrode_positions[neighbor])
            distance = np.linalg.norm(elec_pos - neighbor_pos)
            distances[(elec, neighbor)] = distance
    return distances


def merge_dicts(dict1, dict2, only_overlap=True):
    merged_data = {}
    for key in dict1.keys():
        if key in dict2:
            merged_data[key] = (dict1[key] + dict2[key]) / 2
        else:
            if not only_overlap:
                merged_data[key] = dict1[key]

    if not only_overlap:
        for key in dict2.keys():
            if key not in merged_data:
                merged_data[key] = dict2[key]
    return merged_data


def merge_dicts_weights(dict1, dict2, dict1_weight=0.7, only_overlap=True):
    merged_data = {}
    for key in dict1:
        if key in dict2:
            merged_data[key] = dict1[key] * dict1_weight + dict2[key] * (1 - dict1_weight)
        else:
            if not only_overlap:
                merged_data[key] = dict1[key]

    if not only_overlap:
        for key in dict2.keys():
            if key not in merged_data:
                merged_data[key] = dict2[key]

    return merged_data

import os.path

import numpy as np
import pandas as pd


def get_landmark_from_csv(path):
    df = pd.read_csv(os.path.join(path, "validated_landmark.csv"))

    # Group by landmark and return positions
    positions = {}
    for landmark, group in df.groupby('Landmark'):
        # Filter rows for the current landmark
        landmark_df = df[df['Landmark'] == landmark]

        # Extract x, y, z values and convert to 1x3 array
        pos_array = landmark_df[['x', 'y', 'z']].values.astype(np.float32)[0]

        # Store the 1x3 array for the landmark
        positions[landmark] = pos_array

    return positions


def read_fpz_locations(path):
    df = pd.read_csv(os.path.join(path, "verified_electrode_positions.csv"))
    # Filter the DataFrame for the row where the "Electrodes" column matches "FPz"
    fpz_row = df[df['Electrode'] == 'FPz']

    fpz_coordinates = fpz_row.loc[:, ['x', 'y', 'z']].values.flatten()

    # Convert to a NumPy array if it's not already
    fpz_coordinates_array = np.array(fpz_coordinates)

    return fpz_coordinates_array


def compare_fpz(fpz_verified, fpz_found):
    distance = np.linalg.norm(fpz_verified-fpz_found)

    if distance > 0.005:
        return False
    else:
        return True



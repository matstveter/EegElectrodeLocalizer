import numpy as np
from matplotlib import pyplot as plt

from pos_3d.localizers.base_localizer import BaseLocalizer
from pos_3d.utils.helper_functions import normalize_vector, get_rotation_matrix


class TemplateNzFpzAligner(BaseLocalizer):
    def __init__(self, rotation_matrix, orientation_dictionary, nasion, inion, lpa, rpa, origo,
                 suggested_electrode_positions, template_dict, plot=False, **kwargs):
        super().__init__(rotation_matrix=rotation_matrix, orientation_dictionary=orientation_dictionary,
                         nasion=nasion, inion=inion, lpa=lpa, rpa=rpa, origo=origo, **kwargs)
        self._suggested_electrode_position = suggested_electrode_positions
        self._template_dict = template_dict
        self._aligned_template_dict = None
        self._plot = plot

    @property
    def aligned_template_dict(self):
        if self._aligned_template_dict is not None:
            return self._aligned_template_dict
        else:
            raise ValueError('Aligned template dict is not initialized, missing function call to align template!')

    def _plot_3d(self, templ_array, plot_nasion=True, plot_centroid=False, line1=None, line2=None, temp_nasion=None):
        """
        Plots two arrays as 3D scatter plots with different colors and labels.

        Parameters:
        - array1: The first array to plot, labeled as 'Estimated'.
        """
        fig = plt.figure(figsize=(90, 45))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the first array
        ax.scatter(templ_array[:, 0], templ_array[:, 1], templ_array[:, 2], color='r', label='Template')

        # Plot the second array
        ax.scatter(self._suggested_electrode_position[:, 0],
                   self._suggested_electrode_position[:, 1],
                   self._suggested_electrode_position[:, 2],
                   color='b',
                   label='Estimated')

        if plot_nasion:
            ax.scatter(*self._nasion, label="Nasion", color='black')
            if temp_nasion is not None:
                ax.scatter(*temp_nasion, label="Template Nasion", color='black')

        if plot_centroid:
            ax.scatter(*np.mean(self._suggested_electrode_position, axis=0), label="Centroid", color='y')

        if line1 is not None and len(line1) > 1:
            ax.plot([line1[0][0], line1[1][0]],
                    [line1[0][1], line1[1][1]],
                    [line1[0][2], line1[1][2]], color='black')

        if line2 is not None and len(line2) > 1:
            ax.plot([line2[0][0], line2[1][0]],
                    [line2[0][1], line2[1][1]],
                    [line2[0][2], line2[1][2]], color='black')

        # Set labels for the axes
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        # Set title and legend
        ax.set_title(f'3D Scatter Plot: {self._subject_id}')
        ax.legend()

        # Show the plot
        plt.show()
        plt.close()

    def _update_template_dict(self, aligned_template):
        """ Function that creates a new template dict based on the aligned template

        Parameters
        ----------
        aligned_template: np.ndarray
            the aligned template

        Returns
        -------
        dict
            Template with electrode name as key and 3d pos as value
        """
        electrode_keys = list(self._template_dict.keys())

        temp_dict = {}
        for i, key in enumerate(electrode_keys):
            temp_dict[key] = aligned_template[i]

        return temp_dict

    def _shift_points(self, template_points):
        """ Function which shifts and aligns the centroid of the template to the suggested electrode positions

        Parameters
        ----------
        template_points: np.ndarray
            3D points

        Returns
        -------
        list, np.ndarray()
            shifted template positions, centroid of shifted template
        """
        template_centroid = np.mean(np.array(template_points), axis=0)
        suggested_centroid = np.mean(self._suggested_electrode_position.copy(), axis=0)
        shift_vector = suggested_centroid - template_centroid

        shifted_template = template_points + shift_vector
        if self._plot:
            self._plot_3d(templ_array=shifted_template, plot_centroid=True)
        return shifted_template, suggested_centroid

    def _get_temp_data(self):
        """ Simple function that extracts the values from the template dict, and also extracts the index of nasion

        Returns
        -------
        list, int
            template pos 3d, index of nasion

        """
        electrode_keys = list(self._template_dict.keys())
        nz_index = electrode_keys.index("Nz")
        template_pos = [*self._template_dict.values()]

        return template_pos, nz_index

    def _get_rotation(self, template_points, index_of_nasion, centroid, axis, normalize=True):
        """ Function to rotate the template points around an axis

        Parameters
        ----------
        template_points: np.ndarray
            Template positions in 3D space
        index_of_nasion: int
            index of the nasion point in the template_points
        centroid: np.ndarray
            3D point
        axis: int
            Which axis to rotate around
        normalize: bool
            default True, normalize the vectors

        Returns
        -------
        np.ndarray
            Return the rotated template points
        """
        # Create vectors in both the template and the suggested points
        nasion_centroid_vector = self._nasion - centroid
        template_nasion_centroid_vector = template_points[index_of_nasion] - centroid

        # If the vector should be normalized
        if normalize:
            nasion_centroid_vector = normalize_vector(nasion_centroid_vector)
            template_nasion_centroid_vector = normalize_vector(template_nasion_centroid_vector)

        # Define which axis the angle should be calculated from
        if axis == 2:
            ax0, ax1 = 1, 0
        elif axis == 1:
            ax0, ax1 = 2, 0
        else:
            ax0, ax1 = 2, 1

        # Find the rotation angle
        rotation_angle = (np.arctan2(nasion_centroid_vector[ax0], nasion_centroid_vector[ax1]) -
                          np.arctan2(template_nasion_centroid_vector[ax0], template_nasion_centroid_vector[ax1]))
        # Normalize the rotation angle between [pi, -pi]
        rotation_angle = (rotation_angle + np.pi) % (2 * np.pi) - np.pi
        # Get the rotation matrix
        rotation_matrix = get_rotation_matrix(axis=axis, angle=rotation_angle)

        # Translate the template positions to the origin for rotation
        translated_template_pos = template_points - centroid
        # Rotate around the Z-axis
        rotated_translated_template = rotation_matrix @ translated_template_pos.T
        # Translate the positions back to the original centroid
        rotated_template = rotated_translated_template.T + centroid

        if self._plot:
            self._plot_3d(templ_array=rotated_template, plot_centroid=True, line1=[self._nasion, centroid],
                          line2=[rotated_template[index_of_nasion], centroid],
                          temp_nasion=rotated_template[index_of_nasion])
        return rotated_template

    def _align_template(self):
        """ Function to align the template to the potential electrode positions, based mostly on the nasion points and
        the centroid of the point clouds.

        1. Get the template 3D points and the index of the nasion point
        2. Shift the template positions so that it shares the centroid with the suggested points
        3. Rotate the points around the z axis
        4. Rotate the points around the x-axis
        5. Return the shifted template positions

        Returns
        -------
        np.ndarray
            The shifted and rotated template positions, no scaling is done (should be in meters)
        """
        template_pos, nz_index = self._get_temp_data()
        shifted_template, suggested_centroid = self._shift_points(template_points=template_pos)
        rotated_template = self._get_rotation(template_points=shifted_template,
                                              index_of_nasion=nz_index,
                                              centroid=suggested_centroid,
                                              axis=2)
        # Rotation around x-axis.
        rotated_template = self._get_rotation(template_points=rotated_template,
                                              index_of_nasion=nz_index,
                                              centroid=suggested_centroid,
                                              axis=0,
                                              normalize=False)
        return rotated_template

    def localize_electrodes(self):
        aligned_template = self._align_template()
        self._aligned_template_dict = self._update_template_dict(aligned_template=aligned_template)


class TemplateMidlineAligner:

    def __init__(self, verified_electrodes, template_dict, plot_results=False):
        self._verified_electrodes = verified_electrodes
        self._template_dict = template_dict
        self._plot_results = plot_results

        self._aligned_template = None

    @property
    def aligned_template(self):
        return self._aligned_template

    @staticmethod
    def _plot_arr(template_dict, midline_dict, template_array, midline_array):
        # Prepare the plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot original template points
        ax.scatter(midline_array[:, 0], midline_array[:, 1], midline_array[:, 2],
                   color='blue', label='Midline')

        # Plot transformed template points
        ax.scatter(template_array[:, 0], template_array[:, 1], template_array[:, 2],
                   color='red', label='Transformed Template')

        for label, point in template_dict.items():
            ax.text(point[0], point[1], point[2], label, color='red')

        for label, point in midline_dict.items():
            ax.text(point[0], point[1], point[2], label, color='green')

        # Adding labels and title
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('Procrustes Analysis: Point Clouds Before and After Transformation')
        ax.legend()

        # Show plot
        plt.show()

    def _get_subset(self):
        """ Function that returns the subset of the template dictionary based on the verified template dict

        Returns
        -------
        dict:
            sub dict of the template dictionary

        """
        template_subset = {key: self._template_dict[key] for key in self._verified_electrodes}
        return template_subset

    @staticmethod
    def _calculate_rotation_matrix(translated_template_points, verified_midline):
        """
        Calculate the rotation matrix needed to align the translated template points with the verified midline.
        """
        # Calculate the covariance matrix
        H = np.dot((translated_template_points - verified_midline.mean(axis=0)).T,
                   (verified_midline - verified_midline.mean(axis=0)))

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        rotation_matrix = np.dot(U, Vt)

        # Ensure a right-handed coordinate system
        if np.linalg.det(rotation_matrix) < 0:
            Vt[-1, :] *= -1
            rotation_matrix = np.dot(U, Vt)

        return rotation_matrix

    @staticmethod
    def _perform_shift(arr, shift_vector):
        """ Performs a shift on an array

        Parameters
        ----------
        arr: np.ndarray
            points in 3D space
        shift_vector: np.ndarray
            Shift vector, usually from the mean of two arrays

        Returns
        -------
        np.ndarray:
            shifted vector
        """
        return arr + shift_vector

    @staticmethod
    def _perform_rotation(arr, rot_mat, centroid):
        """ Perform a rotation on an array, subtracts and add the centroid before and after rotation

        Parameters
        ----------
        arr: np.ndarray
            electrode positions in 3d space
        rot_mat: np.ndarray
            rotation matrix
        centroid: np.ndarray
            centroid of the point cloud

        Returns
        -------
        np.ndarray:
            rotated array
        """
        # Apply the rotation
        arr -= centroid
        rotated_template_points = np.dot(arr, rot_mat)
        rotated_template_points += centroid

        return rotated_template_points
        
    def _align_point_clouds(self, plot_results):
        """
        Aligns the template point cloud to the verified midline point cloud.

        This method aligns a subset of the template point cloud to the verified midline
        by first translating the template so its centroid matches the centroid of the verified midline,
        and then applying a rotation matrix to align the orientations of the point clouds.

        Parameters
        ----------
        plot_results: bool
            plot the rotation

        Returns
        -------
        dict
            A dictionary containing the newly aligned template points. Each key corresponds
            to a point label, and the value is a numpy array representing the coordinates of the
            aligned point.

        Notes
        -----
        The method relies on the _perform_shift and _calculate_rotation_matrix methods
        to compute the translation and rotation necessary for alignment. It then applies
        these transformations to the full set of template points.
        """
        template_sub_array = np.array(list(self._get_subset().values()))
        verified_midline = np.array(list(self._verified_electrodes.values()))

        # Translation
        midline_centroid = np.mean(verified_midline, axis=0)
        template_centroid = np.mean(template_sub_array, axis=0)
        translation_vector = midline_centroid - template_centroid

        # Find shift and rotation
        shifted_template = self._perform_shift(template_sub_array, shift_vector=translation_vector)
        rotation_matrix = self._calculate_rotation_matrix(shifted_template, verified_midline)

        # Apply the same transformation to the full template
        full_template_array = np.array(list(self._template_dict.values()))

        shifted_full_template = self._perform_shift(arr=full_template_array, shift_vector=translation_vector)
        rotated_full_template = self._perform_rotation(arr=shifted_full_template, rot_mat=rotation_matrix,
                                                       centroid=midline_centroid)
        newly_aligned_template = {key: val for key, val in zip(self._template_dict, rotated_full_template)}

        if plot_results:
            self._plot_arr(template_dict=newly_aligned_template,
                           template_array=np.array(list(newly_aligned_template.values())),
                           midline_dict=self._verified_electrodes,
                           midline_array=verified_midline)

        return newly_aligned_template

    def align_template(self):
        self._aligned_template = self._align_point_clouds(plot_results=self._plot_results)

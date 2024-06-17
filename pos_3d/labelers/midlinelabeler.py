import numpy as np

from pos_3d.localizers.base_localizer import BaseLocalizer
from pos_3d.labelers.template_aligner import TemplateNzFpzAligner
from pos_3d.utils.helper_functions import cpd, find_electrode_neighbors, get_sub_dict_from_single_key, \
    get_two_key_midline_sub_dict, icp, update_dictionary

from pos_3d.utils.mesh_helper_functions import find_white_gray_vertices


class MidlineLabeler(BaseLocalizer):

    def __init__(self, rotation_matrix, orientation_dictionary, nasion, inion, lpa, rpa, origo,
                 suggested_electrode_positions_rgb, suggested_electrode_positions_hsv, template_dict, fpz, **kwargs):
        super().__init__(rotation_matrix=rotation_matrix, orientation_dictionary=orientation_dictionary,
                         nasion=nasion, inion=inion, lpa=lpa, rpa=rpa, origo=origo, **kwargs)

        self._template_dict = template_dict
        self._fpz = fpz
        self._kwargs = kwargs
        self._temp_mesh = self._get_mesh()

        self._suggested_electrodes_rgb = suggested_electrode_positions_rgb
        self._suggested_electrodes_hsv = suggested_electrode_positions_hsv

        # Where to save the finished product
        self._midline, self._estimated, self._found_rgb, self._found_hsv = None, None, None, None

    @property
    def midline(self):
        if self._midline is not None:
            return self._midline
        else:
            raise ValueError("Midline electrode is None, make sure that the localize function has been called!")

    @property
    def estimated(self):
        if self._estimated is not None:
            return self._estimated
        else:
            raise ValueError("Estimated electrode is None, make sure that the localize function has been called!")

    @property
    def found_hsv(self):
        if self._found_hsv is not None:
            return self._found_hsv
        else:
            raise ValueError("Found HSV is None, make sure that the localize function has been called!")

    @property
    def found_rgb(self):
        if self._found_rgb is not None:
            return self._found_rgb
        else:
            raise ValueError("Found RGB is None, make sure that the localize function has been called!")

    def _do_template_alignment(self, suggested_electrode_positions):
        """ Function that calls the method and class of Template Alignment for aligning the template to the
        suggested electrodes.

        Returns
        -------
        dict
            transformed dictionary containing the newly aligned template positions
        """
        temp_aligner = TemplateNzFpzAligner(orientation_dictionary=self._orientation_dictionary, origo=self._origo,
                                            rotation_matrix=self._rotation_matrix, nasion=self._nasion,
                                            inion=self._inion,
                                            suggested_electrode_positions=suggested_electrode_positions,
                                            template_dict=self._template_dict.copy(),
                                            lpa=self._lpa, rpa=self._rpa, **self._kwargs)
        temp_aligner.localize_electrodes()
        aligned_template = temp_aligner.aligned_template_dict
        del aligned_template['Nz']
        return aligned_template

    def _check_fpz(self, electrode_dict, found_electrode_positions):
        """
        Check for the presence of the 'FPz' key in `electrode_dict`. If 'FPz' exists,
        its value is compared with `self._fpz`. If the Euclidean distance between these two
        points is greater than 0.005, the value in the dictionary is updated to `self._fpz`.
        If 'FPz' does not exist in the dictionary, it is added with the value of `self._fpz`.

        Parameters
        ----------
        electrode_dict : dict
            A dictionary containing electrode positions. Keys are electrode labels
            (e.g., 'FPz'), and values are their corresponding position data.

        Returns
        -------
        dict
            The updated `electrode_dict` with the 'FPz' key checked and set or updated
            as necessary.

        Notes
        -----
        - The method assumes `self._fpz` is a predefined attribute containing the
          reference position data for the 'FPz' electrode.
        - The comparison between `electrode_dict['FPz']` and `self._fpz` is based on
          Euclidean distance.
        - A threshold of 0.005 is used to determine if the value in `electrode_dict`
          needs to be updated to `self._fpz`.
        """

        if "FPz" in electrode_dict:
            index_of_fpz = np.where(np.all(found_electrode_positions == electrode_dict['FPz'], axis=1))[0]
            if np.linalg.norm(electrode_dict["FPz"] - self._fpz) > 0.005:
                electrode_dict["FPz"] = self._fpz
                found_electrode_positions[index_of_fpz] = electrode_dict["FPz"]
        else:
            electrode_dict["FPz"] = self._fpz
            found_electrode_positions = np.append(found_electrode_positions, [self._fpz], axis=0)

        return electrode_dict, found_electrode_positions

    def _align_and_label(self, suggested_positions, use_cpd_first=False):
        """
        Align and label electrode positions.

        This method aligns a template of electrode positions with a set of suggested positions,
        and then labels the electrodes. It employs a sequence of alignment methods: template
        alignment, Coherent Point Drift (CPD), and Iterative Closest Point (ICP). Optionally,
        CPD can be used as the first alignment step. Finally, the method checks and adjusts
        the position of the Fpz electrode.

        Parameters:
        suggested_positions (dict): A dictionary of suggested electrode positions.
        use_cpd_first (bool): If True, use CPD alignment before ICP alignment. Default is False.

        Returns:
        tuple: A tuple containing:
            - aligned_template (dict): The aligned template of electrode positions.
            - labeled_electrode_dict (dict): A dictionary with labeled electrode positions.
            - suggested_positions (dict): The adjusted suggested electrode positions.
        """
        # Perform initial template alignment with the suggested electrode positions.
        aligned_template_dict = self._do_template_alignment(suggested_electrode_positions=suggested_positions.copy())

        # If enabled, refine the alignment using Coherent Point Drift (CPD).
        if use_cpd_first:
            aligned_template_dict = cpd(aligned_template_dict=aligned_template_dict,
                                        suggested_electrode_positions=suggested_positions.copy())

        # Further refine the alignment and label the electrodes using Iterative Closest Point (ICP).
        labeled_electrode_dict, aligned_template = icp(aligned_template_dict=aligned_template_dict,
                                                       suggested_electrode_positions=suggested_positions.copy())

        # Check and adjust the Fpz electrode position, if necessary.
        labeled_electrode_dict, suggested_positions = self._check_fpz(electrode_dict=labeled_electrode_dict,
                                                                      found_electrode_positions=suggested_positions)

        return aligned_template, labeled_electrode_dict, suggested_positions

    def _estimate_electrode(self, key, found_electrodes_dict, neighbors, temp_pos, midline):
        """ This function estimates the electrode based on the neighbors.

        1. Loop through the neighbor dict, and check if the neighbors exists in the estimated electrodes, if it
        exists we use the neighbors to estimate the positions. If no neighbors exist, use the temp pos as a
        potential suggested position.
        2. Use ray-tracing for the point to find candidates that are located on the mesh and use the closest point, if
        ray tracing did not find any points, continue to use the temp_pos as a potential point
        3. Use the function find_white_gray_vertices to find potential candidates around the suggested points,
        because it has to be white/gray if it is an electrode
        4. Return the estimated position

        Parameters
        ----------
        key: str
            the electrode that should be estimated
        found_electrodes_dict: dict
            dictionary containing the values of suggested points from the algorithm
        neighbors: dict
            dictionary containing the distance and direction to the key from its neighbors
        temp_pos: np.ndarray
            3d position from the aligned template, which should be an okay guess if other methods fail

        Returns
        -------
        np.ndarray:
            position that is estimated

        """

        # Initialize a zero vector for the estimated position of the electrode
        estimated_pos = np.zeros(3)
        total_weight = 0

        # Iterate through each neighbor of the missing electrode
        for n_key, n_val in neighbors[key].items():
            # Check if this neighbor's position is known
            if n_key in found_electrodes_dict:
                # Get the known position of the neighbor
                neigh_pos = found_electrodes_dict[n_key]

                # Get the distance from the neighbor to the missing electrode
                dist = n_val['dist']

                # Get the unit direction vector from the neighbor to the missing electrode
                direction = n_val['unit_direction']

                # Calculate a position estimate based on this neighbor's position,
                # the distance, and the direction to the missing electrode
                position_estimate = neigh_pos + direction * dist

                # Calculate the weight for averaging, using the inverse of the distance
                # (closer neighbors have a higher weight)
                weight = 1 / dist

                # Accumulate the weighted position estimate
                estimated_pos += position_estimate * weight

                # Accumulate the total weight
                total_weight += weight

        # If the total weight is non-zero, normalize the estimated position
        # by the total weight to get the average
        if total_weight != 0:
            estimated_pos /= total_weight
        else:
            # If no valid neighbors were found (total weight is zero),
            # use a temporary position (could be a default or fallback position)
            estimated_pos = temp_pos

        # Calculate directions for ray-tracing
        direction_to_centroid = self._temp_mesh.centroid - estimated_pos
        direction_away_from_centroid = -direction_to_centroid

        # Perform ray-tracing
        intersection_points = self._temp_mesh.ray.intersects_location(
            ray_origins=[estimated_pos, estimated_pos],
            ray_directions=[direction_to_centroid, direction_away_from_centroid]
        )[0]  # Assuming this extracts the intersection points from the results

        # Check if there are any intersection points
        if len(intersection_points) > 0:
            # Calculate distances from the estimated position to each intersection point
            dists_to_estimated_point = [np.linalg.norm(estimated_pos - e) for e in intersection_points]

            # Find the index of the minimum distance
            min_distance_index = np.argmin(dists_to_estimated_point)

            # Select the closest intersection point
            estimated_pos = intersection_points[min_distance_index]
        else:
            # If there are no intersection points, keep the original estimated position
            estimated_pos = estimated_pos

        centroids = find_white_gray_vertices(mesh=self._temp_mesh,
                                             vertex_colors=self._get_colors(mesh=self._temp_mesh, rgb=False),
                                             point=estimated_pos,
                                             distance_threshold=0.02)
        if len(centroids) == 0:
            return estimated_pos
        elif len(centroids) == 1:
            return centroids[0]
        else:
            if midline == "center":
                dist_weight = 1
                diff = [np.abs(estimated_pos[0] - centre[0]) for centre in centroids]
                dists = [np.linalg.norm(estimated_pos - centre) for centre in centroids]

                combined_scores = [d + dist_weight * dist for d, dist in zip(diff, dists)]
                return centroids[combined_scores.index(min(combined_scores))]
            else:
                dists = [np.linalg.norm(estimated_pos - centre) for centre in centroids]
                return centroids[dists.index(min(dists))]

    def _verify_center_position(self, key, found_electrodes_dict, neighbor_dict, icp_temp_pos, prev_key, midline,
                                plot_verification=False, distance_between_estimate_and_suggestion=0.02,
                                print_progress=False):
        """ Verifies the position of an electrode based on the template and neighbors.
        
        1. Estimate the position using the function estimate_electrode, this will be a good estimate where it should
        be based on the neighbors. 
        2. Calculate the distance between the found_position which is suggested_electrode_dict[key] and the newly
        estimated position. 
        3. If the distance between the estimated position and found position is more than 
        distance_between_estimate_and_suggestion, it could suggest that the electrode is placed somewhere wrong. If
        below, return the found position.
        4. If the distance is larger -Check distance from nasion for both estimated point, found_point and the template
            4.1 If the distance for the found_point is smaller than 2.5 cm and the distance for the estimated is larger
            return the found, inversely, return the estimated.
            4.2 If both is below, return the closest point to the x-axis of nasion
            4.3 If both is above, check distance for the previous neighbor and relation in relation to the template and
            return the one that is closest.
        
        Parameters
        ----------
        key: str
            label of the electrode that is verified
        found_electrodes_dict: dict
            format {'label': [x, y, z]} for the found electrodes
        neighbor_dict: dict
            dictionary of the neighbors, so the neighbors for the current key is neighbors[key]
        icp_temp_pos: np.ndarray
            Suggested position from the icp aligned template
        midline: str    
            center if the z electrodes is estimated, other if the Cz line to T7 T8
        prev_key: str
            key of the previous electrode
        plot_verification: bool
            plot estimated and true values
        distance_between_estimate_and_suggestion: float
            the distance measure between the estimated and true, if that is large, do more analysis
        print_progress: bool
            If print during the run
        

        Returns
        -------

        """
        was_estimated = False
        estimated_pos = self._estimate_electrode(key=key,
                                                 found_electrodes_dict=found_electrodes_dict,
                                                 neighbors=neighbor_dict,
                                                 temp_pos=icp_temp_pos,
                                                 midline=midline)
        if plot_verification:
            self._plot_mesh(rotation_matrix=self._rotation_matrix,
                            points=[estimated_pos, found_electrodes_dict[key]],
                            labels=['Est', 'True'])
        dist_between = np.linalg.norm(estimated_pos - found_electrodes_dict[key])

        if dist_between > distance_between_estimate_and_suggestion:
            if prev_key is not None and prev_key in found_electrodes_dict:
                dist_from_previous_electrode_found = np.linalg.norm(found_electrodes_dict[prev_key] -
                                                                    found_electrodes_dict[key])
                dist_from_previous_electrode_estimated = np.linalg.norm(found_electrodes_dict[prev_key] -
                                                                        estimated_pos)
                template_dist = neighbor_dict[key][prev_key]['dist']

                diff_temp_dist_found = np.abs(template_dist - dist_from_previous_electrode_found)
                diff_temp_dist_estimated = np.abs(template_dist - dist_from_previous_electrode_estimated)
            else:
                diff_temp_dist_found, diff_temp_dist_estimated = 0.02, 0.02

            # If one of them is above 0.2 centimeter in difference from the template
            if diff_temp_dist_found > 0.01 > diff_temp_dist_estimated:
                position = estimated_pos
                was_estimated = True
            elif diff_temp_dist_found < 0.01 < diff_temp_dist_estimated:
                position = found_electrodes_dict[key]
            else:
                if midline == "center":
                    # Distance from nasion to the estimated point, should be somewhat similar to the template
                    dist_nz_sug = np.linalg.norm(self._nasion - found_electrodes_dict[key])
                    dist_nz_est = np.linalg.norm(self._nasion - estimated_pos)
                    dist_nz_temp = np.linalg.norm(self._nasion - icp_temp_pos)

                    diff_from_temp_suggested = np.abs(dist_nz_temp - dist_nz_sug)
                    diff_from_temp_est = np.abs(dist_nz_temp - dist_nz_est)

                    # If the found point is below and the estimated position is above, return the found
                    if diff_from_temp_suggested < 0.025 < diff_from_temp_est:
                        position = found_electrodes_dict[key]
                    # Return the estimated if that is a better suggestions
                    elif diff_from_temp_suggested > 0.025 > diff_from_temp_est:
                        position = estimated_pos
                        was_estimated = True
                    # Both are potential candidates, return the one that is most closely to the nasion x value
                    elif diff_from_temp_suggested < 0.025 > diff_from_temp_est:
                        diff_x_sug = np.abs(self._nasion[0] - found_electrodes_dict[key][0])
                        diff_x_est = np.abs(self._nasion[0] - estimated_pos[0])

                        if diff_x_est > diff_x_sug:
                            position = found_electrodes_dict[key]
                        else:
                            was_estimated = True
                            position = estimated_pos
                    # If no good candidates return the one that is based on some rules in the template
                    else:
                        was_estimated = True
                        position = estimated_pos

                else:
                    # Distance from nasion to the estimated point, should be somewhat similar to the template
                    dist_nz_sug = np.linalg.norm(self._nasion - found_electrodes_dict[key])
                    dist_nz_est = np.linalg.norm(self._nasion - estimated_pos)
                    dist_nz_temp = np.linalg.norm(self._nasion - icp_temp_pos)

                    diff_from_temp_suggested = np.abs(dist_nz_temp - dist_nz_sug)
                    diff_from_temp_est = np.abs(dist_nz_temp - dist_nz_est)

                    # If the found point is below and the estimated position is above, return the found
                    if diff_from_temp_suggested < 0.025 < diff_from_temp_est:
                        position = found_electrodes_dict[key]
                    # Return the estimated if that is a better suggestions
                    elif diff_from_temp_suggested > 0.025 > diff_from_temp_est:
                        position = estimated_pos
                        was_estimated = True
                    elif diff_from_temp_suggested < 0.025 > diff_from_temp_est:
                        diff_x_sug = np.abs(found_electrodes_dict['Cz'][1] - found_electrodes_dict[key][1])
                        diff_x_est = np.abs(found_electrodes_dict['Cz'][1] - estimated_pos[1])

                        if diff_x_est > diff_x_sug:
                            position = found_electrodes_dict[key]
                        else:
                            was_estimated = True
                            position = estimated_pos
                    # If no good candidates return the one that is based on some rules in the template
                    else:
                        was_estimated = True
                        position = estimated_pos
        else:
            position = found_electrodes_dict[key]

        if print_progress:
            if was_estimated:
                print(f"Position was estimated: {key}")
            else:
                print(f"Using the found position: {key}")

        return position, was_estimated

    def _verify_center_midline(self, found_electrodes, icp_aligned_template, print_progress=False):
        """ This function is verifying the electrodes in the midline.

        This function will loop through the template, and see if all the same keys and electrodes exists in the
        suggested electrode dictionary. If it does exist, it is verified in a seperate function, to make sure that
        the suggested position is correct, if it does not exist, it is estimated based on neighbors from the
        aligned template.

        Parameters
        ----------
        found_electrodes: dict
            structure of the dictionary is like this: {electrode_name: [x, y, z]}
        icp_aligned_template: dict
            same structure as the suggested electrodes, but here all positions are from an aligned template, so
            can be used to give a reasonable estimate on the positions

        Returns
        -------
        [dict, dict]:
            dictionary containing the midline electrodes that are found and estimated and only the estimated electrodes

        """
        # Get the electrodes from the template that has z in the label, and sort based on the x-value
        template_midline = dict(
            sorted(get_sub_dict_from_single_key(dictionary=self._template_dict.copy(), key="z").items(),
                   key=lambda item: item[1][0], reverse=True))

        # Get the midline electrodes from the aligned template and crate a neighbor dict with those aligned values
        icp_midline_dict = get_sub_dict_from_single_key(dictionary=icp_aligned_template.copy(), key="z")
        neighbors_dict = find_electrode_neighbors(electrode_dict=icp_midline_dict, radius=0.075)

        # Get the midline electrodes from the estimated electrode positions
        estimated_electrode_midline_dict = get_sub_dict_from_single_key(dictionary=found_electrodes.copy(), key="z")

        # Set values
        prev_key = None
        estimated_positions = {}
        # Set the nasion and fpz values in the new midline dictionary
        midline_electrodes = {'Nz': self._nasion,
                              'FPz': estimated_electrode_midline_dict['FPz']}

        # Loop through the template keys
        for key in template_midline:
            estimated = False
            # FPz is already verified!
            if key == "Nz" or key == "FPz":
                if print_progress:
                    print(f"Position verified and okay: {key}")
                prev_key = key
                continue

            if print_progress:
                print(f"\n\nWorking on key: {key}")

            # If the current key is not within the found estimated electrodes, we have to estimate this position
            if key not in estimated_electrode_midline_dict:
                if print_progress:
                    print("Estimating....")
                position = self._estimate_electrode(key=key,
                                                    found_electrodes_dict=estimated_electrode_midline_dict,
                                                    neighbors=neighbors_dict,
                                                    temp_pos=icp_midline_dict[key],
                                                    midline="center")
                estimated = True
            else:
                if print_progress:
                    print("Verifying....")
                # IF the key does exist, we want to verify that it indeed is correctly placed.
                position, estimated = (
                    self._verify_center_position(key=key,
                                                 found_electrodes_dict=estimated_electrode_midline_dict,
                                                 neighbor_dict=neighbors_dict,
                                                 icp_temp_pos=icp_midline_dict[key],
                                                 prev_key=prev_key, midline="center"))
            # If the current electrode is estimated, save this is a dictionary
            if estimated:
                estimated_positions[key] = position

            estimated_electrode_midline_dict[key] = position
            # Set the found value to the midline dict
            midline_electrodes[key] = position
            prev_key = key

        return midline_electrodes, estimated_positions

    def _verify_other_midline(self, found_electrodes, icp_aligned_template, print_progress=False):
        other_midline_electrodes, estimated_electrodes = {}, {}

        # Get the electrodes from the template that has z in the label, and sort based on the x-value
        template_midline = dict(
            sorted(get_two_key_midline_sub_dict(dictionary=self._template_dict.copy()).items(),
                   key=lambda item: item[1][1], reverse=True))
        # Get the midline electrodes from the aligned template and crate a neighbor dict with those aligned values
        icp_midline_dict = get_two_key_midline_sub_dict(dictionary=icp_aligned_template.copy())
        neighbors_dict = find_electrode_neighbors(electrode_dict=icp_midline_dict, radius=0.075)
        found_electrode_midline_dict = get_two_key_midline_sub_dict(dictionary=found_electrodes.copy())

        prev_key = None

        for key in template_midline:
            estimated = False
            # Verified already, but kept to help estimate other electrodes
            if key == "Cz":
                prev_key = key
                continue

            if print_progress:
                print(f"\n\nWorking on key: {key}")

            if key not in found_electrode_midline_dict:
                if print_progress:
                    print("Estimating....")
                position = self._estimate_electrode(key=key,
                                                    found_electrodes_dict=found_electrode_midline_dict,
                                                    neighbors=neighbors_dict,
                                                    temp_pos=icp_midline_dict[key],
                                                    midline="other")
                estimated = True

                # todo Find out if we should add the point estimated to the found_electrode_midline_dict???
                # found_electrode_midline_dict[key] = position
            else:
                if print_progress:
                    print("Verifying....")
                position, estimated = (
                    self._verify_center_position(key=key,
                                                 found_electrodes_dict=found_electrode_midline_dict,
                                                 neighbor_dict=neighbors_dict,
                                                 icp_temp_pos=icp_midline_dict[key],
                                                 prev_key=prev_key, midline="other", plot_verification=False,
                                                 print_progress=print_progress))

            if estimated:
                estimated_electrodes[key] = position
            other_midline_electrodes[key] = position
            prev_key = key

        return other_midline_electrodes, estimated_electrodes

    def _get_midline_electrodes(self, electrodes_rgb, electrodes_hsv, icp_rgb, icp_hsv, midline, plot=False):
        """ Function to find the midline electrodes using both the rgb and hsv arrays and returns the one where
        the fewest electrode had to be estimated

        Parameters
        ----------
        electrodes_rgb: dict
            the suggested electrodes using rgb
        electrodes_hsv: dict
            the suggested electrode using the hsv colors
        icp_rgb: dict
            Aligned template values for the rgb
        icp_hsv: dict
            Aligned template values for the hsv
        plot: bool
            If true plot the midline electrodes


        Returns
        -------
        [dict, dict]:
            midline electrodes and estimated electrodes

        Notes:
        -------
        - All dictionaries on the form {electrode_key: [x, y, z]}

        """
        if midline == "center":
            # Get the verified and estimated electrode dicts using the rgb values
            midline_dict_rgb, estimated_rgb = self._verify_center_midline(found_electrodes=electrodes_rgb,
                                                                          icp_aligned_template=icp_rgb,
                                                                          print_progress=False)
            # Get the verified and estimated electrode dicts using the hsv values
            midline_dict_hsv, estimated_hsv = self._verify_center_midline(found_electrodes=electrodes_hsv,
                                                                          icp_aligned_template=icp_hsv,
                                                                          print_progress=False)
        else:
            # Get the verified and estimated electrode dicts using the rgb values
            midline_dict_rgb, estimated_rgb = self._verify_other_midline(found_electrodes=electrodes_rgb,
                                                                         icp_aligned_template=icp_rgb,
                                                                         print_progress=False)
            # Get the verified and estimated electrode dicts using the hsv values
            midline_dict_hsv, estimated_hsv = self._verify_other_midline(found_electrodes=electrodes_hsv,
                                                                         icp_aligned_template=icp_hsv,
                                                                         print_progress=False)

        # If the number of estimated keys is larger or similar in rgb, return hsv, else return rgb
        if len(estimated_rgb.keys()) > len(estimated_hsv.keys()):
            if plot:
                self._plot_mesh(rotation_matrix=self._rotation_matrix, points=[*midline_dict_hsv.values()],
                                labels=list(midline_dict_hsv.keys()))
            return midline_dict_hsv, estimated_hsv
        else:
            if plot:
                self._plot_mesh(rotation_matrix=self._rotation_matrix, points=[*midline_dict_rgb.values()],
                                labels=list(midline_dict_rgb.keys()))
            return midline_dict_rgb, estimated_rgb

    def localize_electrodes(self):
        """ Aligns and label the RGB suggested and HSV suggested electrodes, picks the one that is estimating the
        fewest as the method of suggesting from the earlier colors is seen as the most robust.

        Returns
        -------
        None
        """
        icp_aligned_rgb, electrode_dict_rgb, rgb_electrodes = self._align_and_label(
            suggested_positions=self._suggested_electrodes_rgb.copy())
        icp_aligned_hsv, electrode_dict_hsv, hsv_electrodes = self._align_and_label(
            suggested_positions=self._suggested_electrodes_hsv.copy())

        midline_dictionary, estimated_dictionary = self._get_midline_electrodes(
            electrodes_rgb=electrode_dict_rgb.copy(),
            icp_rgb=icp_aligned_rgb,
            electrodes_hsv=electrode_dict_hsv.copy(),
            icp_hsv=icp_aligned_hsv,
            midline="center")

        electrode_dict_rgb = update_dictionary(old_dict=electrode_dict_rgb, new_dict=midline_dictionary)
        electrode_dict_hsv = update_dictionary(old_dict=electrode_dict_hsv, new_dict=midline_dictionary)

        other_finished, other_estimated = self._get_midline_electrodes(electrodes_rgb=electrode_dict_rgb,
                                                                       icp_rgb=icp_aligned_rgb,
                                                                       electrodes_hsv=electrode_dict_hsv,
                                                                       icp_hsv=icp_aligned_hsv,
                                                                       midline="other")

        midline_dictionary = update_dictionary(old_dict=midline_dictionary, new_dict=other_finished)
        # self._plot_mesh(rotation_matrix=self._rotation_matrix, points=[*midline_dictionary.values()],
        #                 labels=list(midline_dictionary))
        estimated_dictionary = update_dictionary(old_dict=estimated_dictionary, new_dict=other_estimated)

        self._found_rgb = self._suggested_electrodes_rgb.copy()
        self._found_hsv = self._suggested_electrodes_hsv.copy()

        # Add the estimated points to the lists
        for v in estimated_dictionary.values():
            self._found_hsv = np.append(self._found_hsv, values=[v], axis=0)
            self._found_rgb = np.append(self._found_rgb, values=[v], axis=0)

        self._midline = midline_dictionary
        self._estimated = estimated_dictionary

    def plot_solution(self):
        if self._midline is not None:
            self._plot_mesh(rotation_matrix=self._rotation_matrix, points=[*self._midline.values()],
                            labels=list(self._midline.keys()))
        else:
            raise ValueError("Function not yet called so the dictionary to plot from is None")

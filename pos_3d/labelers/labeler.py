import numpy as np
from scipy.spatial.distance import cosine

from pos_3d.labelers.template_aligner import TemplateMidlineAligner
from pos_3d.localizers.base_localizer import BaseLocalizer
from pos_3d.utils.helper_functions import calculate_pairwise_distances, find_electrode_neighbors, icp, merge_dicts, \
    merge_dicts_weights, update_dictionary
from pos_3d.utils.mesh_helper_functions import find_white_gray_vertices


class Labeler(BaseLocalizer):

    def __init__(self, rotation_matrix, orientation_dictionary, nasion, inion, lpa, rpa, origo,
                 template_dict, verified_midline, found_rgb, found_hsv, **kwargs):
        super().__init__(rotation_matrix=rotation_matrix, orientation_dictionary=orientation_dictionary,
                         nasion=nasion, inion=inion, lpa=lpa, rpa=rpa, origo=origo, **kwargs)

        self._template_dict = template_dict
        self._kwargs = kwargs
        self._temp_mesh = self._get_mesh()

        self._verified_midline = verified_midline
        self._found_rgb = found_rgb
        self._found_hsv = found_hsv

        self._verified_electrodes = verified_midline.copy()

        self._finished_solution, self._final_estimates = None, None

    @property
    def finished_solution(self):
        return self._finished_solution

    @property
    def final_estimates(self):
        return self._final_estimates

    def plot_solution(self, dict_to_plot=None):
        """ Function to plot a dictionary.

        Parameters
        ----------
        dict_to_plot: dict
            dictionary to plot, defaults to None

        Returns
        -------
        None
        """
        if dict_to_plot is None:
            if self._finished_solution is not None:
                self._plot_mesh(rotation_matrix=self._rotation_matrix, points=[*self._finished_solution.values()],
                                labels=list(self._finished_solution))
            else:
                raise ValueError("No dict supplied and the finished_solution is not set yet!")
        else:
            self._plot_mesh(rotation_matrix=self._rotation_matrix, points=[*dict_to_plot.values()],
                            labels=list(dict_to_plot))

    def _align_template_and_label(self, verified_electrodes, midline=False, plot_results=False):
        """ This function uses the TemplateMidlineAligner class to align the template to the verified midline from
        previous functions. Then this newly aligned template is used to label the found hsv and rgb values.

        Returns
        -------
        [dict, dict, dict, dict]:
            labeled rgb dict, labeled hsv dict, icp aligned template dict
        """

        temp_aligner = TemplateMidlineAligner(verified_electrodes=verified_electrodes,
                                              template_dict=self._template_dict.copy(),
                                              plot_results=plot_results)
        temp_aligner.align_template()
        aligned_template = temp_aligner.aligned_template
        if midline:
            elect_rgb, icp_rgb = icp(aligned_template_dict=aligned_template,
                                     suggested_electrode_positions=self._found_rgb)
            elect_hsv, icp_hsv = icp(aligned_template_dict=aligned_template,
                                     suggested_electrode_positions=self._found_hsv)
            return elect_rgb, elect_hsv, icp_rgb, icp_hsv
        else:
            elect, icp_elect = icp(aligned_template_dict=aligned_template,
                                   suggested_electrode_positions=[*verified_electrodes.values()])
            return elect, icp_elect

    def _estimate_electrode(self, verified_electrodes, verified_neighbors, only_neighbors=False, dist_threshold=0.02):
        """ Estimates the electrode based on the verified neighbors and verified electrodes around.

        1. Estimate the position based on the surrounding electrodes
        2. Ray trace from that point in both directions to/from the center
        3. Choose the point found closest to the origin from the ray tracing
        4. Search for white vertices near that point and select the closest, if none is found, return the prev estimated

        Parameters
        ----------
        verified_electrodes: dict
            the surrounding electrodes around the current electrodes that are already verified
        verified_neighbors: dict
            neighbour dict from the template with distances and directions from the verified electrodes

        Returns
        -------
        np.ndarray:
            the estimated position
        """
        # Initialize a zero vector for the estimated position of the electrode
        estimated_pos = np.zeros(3)
        total_weight = 0

        # Iterate through each neighbor of the missing electrode
        for n_key, n_val in verified_neighbors.items():
            # Get the known position of the neighbor
            neigh_pos = verified_electrodes[n_key]

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
        estimated_pos /= total_weight

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
            print("No intersection")

        if only_neighbors:
            return estimated_pos

        centroids = find_white_gray_vertices(mesh=self._temp_mesh,
                                             vertex_colors=self._get_colors(mesh=self._temp_mesh, rgb=False),
                                             point=estimated_pos,
                                             distance_threshold=dist_threshold)
        if len(centroids) == 0:
            return estimated_pos
        elif len(centroids) == 1:
            return centroids[0]
        else:
            dists = [np.linalg.norm(estimated_pos - centre) for centre in centroids]
            return centroids[dists.index(min(dists))]

    def _verify_electrode(self, found_position, verified_electrodes, verified_neighbors, icp_position, threshold=0.015,
                          plot_verification=False):
        """ Function to verify the position of an electrode.

        1. Estimate the position by using the function _estimated_electrode. Which will give an estimate based on the
        template and already found electrodes
        2. Compare distances from the found electrode to the estimated and to the icp aligned position from the template
            2.1 If either the estimated or icp are close to the found point, return that
            2.2 If icp and estimated is close, return the estimated
            2.3 Else return the icp, but only after searching the surrounding are for vertices, if none is found,
            return the found position

        Parameters
        ----------
        found_position: np.ndarray
            The position found through segmentation and later labeled
        verified_electrodes: dict
            verified electrodes
        verified_neighbors: dict
            verified neighbors to the current electrode that is verified
        icp_position: np.ndarray
            The position found through ICP alignment of the template to the found positions
        threshold: float
            Distances used to choose the point that is most likely to be correct.
        plot_verification: bool,
            defaults to False, plots the estimated, found and icp onn the mesh

        Returns
        -------
        np.ndarray, bool:
            the chosen point, and a boolean saying if the point was estimated or not

        """
        was_estimated = False
        estimated_pos = self._estimate_electrode(verified_electrodes=verified_electrodes,
                                                 verified_neighbors=verified_neighbors)
        if plot_verification:
            self._plot_mesh(rotation_matrix=self._rotation_matrix,
                            points=[estimated_pos, found_position, icp_position],
                            labels=['Est', 'True', 'ICP'])

        # If estimated position is in verified?
        dist_to_estimated = np.linalg.norm(found_position - estimated_pos)
        dist_to_icp = np.linalg.norm(found_position - icp_position)
        dist_icp_est = np.linalg.norm(estimated_pos - icp_position)

        # Decision-making based on the proximity
        if dist_to_estimated <= threshold or dist_to_icp <= threshold:
            # If both are close enough to the found position
            final_position = found_position
        elif dist_icp_est <= threshold:
            # If only estimated is close enough
            final_position = estimated_pos
            was_estimated = True
        else:
            centroids = find_white_gray_vertices(mesh=self._temp_mesh,
                                                 vertex_colors=self._get_colors(mesh=self._temp_mesh, rgb=False),
                                                 point=icp_position,
                                                 distance_threshold=0.01)
            if len(centroids) == 0:
                final_position = found_position
            elif len(centroids) == 1:
                final_position = centroids[0]
                was_estimated = True
            else:
                dists = [np.linalg.norm(estimated_pos - centre) for centre in centroids]
                final_position = centroids[dists.index(min(dists))]
                was_estimated = True

        return final_position, was_estimated

    def _verification(self, icp_aligned_dict, found_and_labeled_electrodes, neighbor_dist=0.04):
        """ This function verifies the found electrodes, and estimates the missing based on the values in the
        template.

        1. Find neighbors to all electrodes based on the neighbor_dist
        2. Find the un-verified electrode, in the beginning that is all except the midline electrodes provided in the
        constructor of this function.
        3. Loop through the unverified electrodes
        4. Find the electrode with the most verified neighbors, and if it is in the found_dictionary, verify else
        estimate the function
        5. Remove the now verified electrode from the unverified list, and iterate until all electrodes are verified or
        estimated
        6. Keep track of which of the electrode is found, and which has to be estimated, this is only for later
        validation

        Parameters
        ----------
        icp_aligned_dict: dict
            Dictionary of the ICP aligned template
        found_and_labeled_electrodes: dict
            The electrodes found through segmentation
        neighbor_dist: float
            the distance in meter that categorize surrounding electrodes as neighbors

        Returns
        -------
        dict, dict:
            verified electrodes, estimated electrodes
        """
        neighbor_dict = find_electrode_neighbors(electrode_dict=icp_aligned_dict, radius=neighbor_dist)
        unverified_electrodes = set(self._template_dict) - set(self._verified_electrodes)

        verified = self._verified_electrodes.copy()
        estimated = {}

        while unverified_electrodes:
            estimated_pos = False
            neighbor_counts = {elec: sum(nb in verified for nb in neighbor_dict[elec])
                               for elec in unverified_electrodes}
            electrode_to_estimate = max(neighbor_counts, key=neighbor_counts.get)
            verified_neighbors = {n_key: neighbor_dict[electrode_to_estimate][n_key]
                                  for n_key in neighbor_dict[electrode_to_estimate]
                                  if n_key in verified}

            if electrode_to_estimate in found_and_labeled_electrodes:
                position, estimated_pos = self._verify_electrode(
                    found_position=found_and_labeled_electrodes[electrode_to_estimate],
                    icp_position=icp_aligned_dict[electrode_to_estimate],
                    verified_electrodes=verified,
                    verified_neighbors=verified_neighbors
                )
            else:
                position = self._estimate_electrode(verified_electrodes=verified,
                                                    verified_neighbors=verified_neighbors)
                estimated_pos = True

            verified[electrode_to_estimate] = position
            unverified_electrodes.remove(electrode_to_estimate)
            if estimated_pos:
                estimated[electrode_to_estimate] = position

        return verified, estimated

    @staticmethod
    def _detect_overlaps(icp_aligned_dict, verified_electrodes, is_overlap=0.0075):
        """ This functon detects overlapping electrodes in the verified electrodes based on a distance measure.

        Parameters
        ----------
        icp_aligned_dict: dict
            icp aligned template dictionary of all electrodes
        verified_electrodes: dict
            verified electrodes from previous algorithms
        is_overlap: float
            distance in meters for being an overlapping electrode

        Returns
        -------
        list, dict :
            list of electrode pair (electrode1, electrode2) that are overlapping,
            dictionary without the overlapping electrode, so
        """
        # Get dictionary for the neighbors
        neighbor_dict = find_electrode_neighbors(electrode_dict=icp_aligned_dict, radius=0.05)
        # Calculate the distance between all electrodes
        verified_electrode_distances = calculate_pairwise_distances(electrode_positions=verified_electrodes,
                                                                    neighbor_dict=neighbor_dict)
        # Calculate the distance between all electrodes in the icp aligned dictionary
        icp_template_dists = calculate_pairwise_distances(electrode_positions=icp_aligned_dict,
                                                          neighbor_dict=neighbor_dict)

        discrepancies = set()
        elec_list = []
        # For every pair in the distances see if the distance is overlapping and that it is over the icp template dist
        for elec_pair, verified_dist in verified_electrode_distances.items():
            icp_distance = icp_template_dists.get(elec_pair)
            if icp_distance and verified_dist < is_overlap < abs(verified_dist - icp_distance):
                # Only keep unique pairs
                unique_pair = tuple(sorted(elec_pair))
                discrepancies.add(unique_pair)

                # Save in list for later sorting of the verified_electrodes
                if elec_pair[0] not in elec_list:
                    elec_list.append(elec_pair[0])

                if elec_pair[1] not in elec_list:
                    elec_list.append(elec_pair[1])

        temp_verified = {key: value for key, value in verified_electrodes.items() if key not in elec_list}
        return discrepancies, temp_verified

    def label_align_fix_solution(self, found_electrodes, icp_dictionary):
        """ This function verifies the found electrodes label dictionary, detects overlapping electrodes and
        then does a new alignment of electrodes, and a new icp.

        Parameters
        ----------
        found_electrodes: dict
            dictionary of the found electrodes on the format {electrode_label: 3d pos}
        icp_dictionary: dict
            dictionary with the aligned icp template dictionary, same format as found_electrodes

        Returns
        -------
        dict, dict, list, dict, dict:
                verified (dict) - verified electrodes
                estimated (dict) - estimated electrodes
                discrep_pair - list of list of a pair of electrodes that are overlapping
                temp_verified (dict) - same as verified, but without the electrode in discrep_pair
                icp_dictionary - icp aligned template

        """
        verified, estimated = self._verification(icp_aligned_dict=icp_dictionary,
                                                 found_and_labeled_electrodes=found_electrodes)
        discrep_pair, temp_verified = self._detect_overlaps(icp_aligned_dict=icp_dictionary,
                                                            verified_electrodes=verified)
        # Perform a new template alignment with all verified electrode, so that the icp is more reliable!
        _, icp_dictionary = (
            self._align_template_and_label(verified_electrodes=temp_verified, plot_results=False))

        return verified, estimated, discrep_pair, temp_verified, icp_dictionary

    def _merge_verified_electrodes(self, verified_rgb, verified_hsv, use_weighting=False, plot_results=False):
        """ Merge the verified electrodes without overlapping, and the option is to use average or to use a 
        weighted average, weighted average is standard
        
        Parameters
        ----------
        verified_rgb: dict
            dictionary of non-overlapping verified electrode
        verified_hsv: dict
            dictionary of non-overlapping verified electrode
        use_weighting: bool
            if true, use a weighted average, default is true, and set to 0.8 weight for RGB
        plot_results: bool
            if true plot the merged dictionary, default is False

        Returns
        -------
        dict:
            merged dictionary
        """

        if use_weighting:
            merged = merge_dicts_weights(dict1=verified_rgb, dict2=verified_hsv, dict1_weight=0.80)
        else:
            merged = merge_dicts(dict1=verified_rgb, dict2=verified_hsv)

        if plot_results:
            self.plot_solution(dict_to_plot=merged)

        return merged

    def _handle_overlap(self, electrode_key1, electrode_key2, overlap_rgb, no_overlap_rgb, icp_rgb,
                        overlap_hsv, no_overlap_hsv, icp_hsv):
        """ This function attempts to handle cases where there are overlap of electrodes. This involves finding the
        correct placed electrode of the two overlapping and estimating the position of the other.

        1. Get the points for rgb point1 and point 2, as well as point 1 and point 2 for HSV, and also two versions
        of the estimated positions.
        2. Average the position of the hsv and rgb, as it should be a good estimate
        3. Use the function verify_with_neighbors to evaluate the most likely electrode that is placed correctly
        4. After the initial electrode is chosen, evaluate the two estimated function and choose the best fitting one


        Parameters
        ----------
        electrode_key1: str
        electrode_key2: str
        overlap_rgb: dict
            dictionary with all found electrodes, including the overlapping
        no_overlap_rgb: dict
            dictionary with all the electrodes that are evaluated here removed
        icp_rgb: dict
            icp aligned dictionary w
        overlap_hsv: dict
            dictionary both all found electrodes in hSV, including the overlapping
        no_overlap_hsv: dict
            dictionary with only the electrodes that are not overlapping
        icp_hsv: dict
            icp aligned hsv dict

        Returns
        -------
            np.ndarray, np.ndarray:
                pos1 and pos2 for electrode key1 and electrode key 2
        """
        # Get point, estimated point, and estimated_point with white vertices from the function
        rgb_p_1, rgb_est_p_1, rgb_est_p_1_2 = (
            self._get_points_for_overlap_estimation(key=electrode_key1,
                                                    overlap=overlap_rgb,
                                                    no_overlap=no_overlap_rgb,
                                                    icp_temp=icp_rgb))
        rgb_p_2, rgb_est_p_2, rgb_est_p_2_2 = (
            self._get_points_for_overlap_estimation(key=electrode_key2,
                                                    overlap=overlap_rgb,
                                                    no_overlap=no_overlap_rgb,
                                                    icp_temp=icp_rgb))

        # Get information from both point from RGB
        hsv_p_1, hsv_est_p_1, hsv_est_p_1_2 = (
            self._get_points_for_overlap_estimation(key=electrode_key1,
                                                    overlap=overlap_hsv,
                                                    no_overlap=no_overlap_hsv,
                                                    icp_temp=icp_hsv))
        hsv_p_2, hsv_est_p_2, hsv_est_p_2_2 = (
            self._get_points_for_overlap_estimation(key=electrode_key2,
                                                    overlap=overlap_hsv,
                                                    no_overlap=no_overlap_hsv,
                                                    icp_temp=icp_hsv))

        # The average values should be the best here, should not be many differences here.
        p1_avg = np.mean([rgb_p_1, hsv_p_1], axis=0)
        p2_avg = np.mean([rgb_p_2, hsv_p_2], axis=0)
        p1_avg_est_1 = np.mean([rgb_est_p_1, hsv_est_p_1], axis=0)
        p1_avg_est_2 = np.mean([rgb_est_p_1_2, hsv_est_p_1_2], axis=0)
        p2_avg_est_1 = np.mean([rgb_est_p_2, hsv_est_p_2], axis=0)
        p2_avg_est_2 = np.mean([rgb_est_p_2_2, hsv_est_p_2_2], axis=0)

        # Evaluate p1 pos, and find if it is more closely matching with key 1 or key 2
        similarity_key1 = self._verify_with_neighbors(icp_aligned=icp_hsv, key=electrode_key1,
                                                      pos=p1_avg, other_pos=p2_avg, other_key=electrode_key2,
                                                      copy_of_non_overlapping_dict=no_overlap_hsv.copy())
        similarity_key2 = self._verify_with_neighbors(icp_aligned=icp_hsv, key=electrode_key2,
                                                      other_key=electrode_key1, other_pos=p2_avg,
                                                      pos=p1_avg, copy_of_non_overlapping_dict=no_overlap_hsv.copy())
        # If the similarity for key 1 is larger, the first return value should be p1
        if similarity_key1 > similarity_key2:
            # Evaluate the similarity of the two differently estimated points, and return the most similar
            similarity_est1 = self._verify_with_neighbors(icp_aligned=icp_hsv, key=electrode_key2,
                                                          other_key=electrode_key1, other_pos=p1_avg,
                                                          pos=p2_avg_est_1,
                                                          copy_of_non_overlapping_dict=no_overlap_hsv.copy())
            similarity_est2 = self._verify_with_neighbors(icp_aligned=icp_hsv, key=electrode_key2,
                                                          other_key=electrode_key1, other_pos=p1_avg,
                                                          pos=p2_avg_est_2,
                                                          copy_of_non_overlapping_dict=no_overlap_hsv.copy())
            # If the second estimated is most similar, return that
            if similarity_est2 > similarity_est1:
                return p1_avg, p2_avg_est_2, 1
            else:
                return p1_avg, p2_avg_est_1, 1
        # Similarity is greater for key 2, return p1_avg as key number 2.
        else:
            # Evaluate the similarity of the two differently estimated points, and return the most similar
            similarity_est1 = self._verify_with_neighbors(icp_aligned=icp_hsv, key=electrode_key2,
                                                          other_key=electrode_key2, other_pos=p1_avg,
                                                          pos=p1_avg_est_1,
                                                          copy_of_non_overlapping_dict=no_overlap_hsv.copy())
            similarity_est2 = self._verify_with_neighbors(icp_aligned=icp_hsv, key=electrode_key2,
                                                          other_key=electrode_key2, other_pos=p1_avg,
                                                          pos=p1_avg_est_2,
                                                          copy_of_non_overlapping_dict=no_overlap_hsv.copy())
            if similarity_est2 > similarity_est1:
                return p1_avg_est_2, p1_avg, 0
            else:
                return p1_avg_est_1, p1_avg, 0

    @staticmethod
    def _verify_with_neighbors(icp_aligned, key, pos, copy_of_non_overlapping_dict, other_key=None, other_pos=None):
        """
        Verify the similarity of an electrode's position with its neighbors.

        This function calculates a similarity score based on the distance and directional alignment
        between an electrode's position and its neighbors. It considers both the actual position
        from `icp_aligned` data and a hypothetical position `pos`. The score is an average of the
        individual similarity measures for each neighbor.

        Parameters
        ----------
        icp_aligned : dict
            A dictionary containing ICP-aligned data of electrodes.
        key : str
            The key identifying the electrode whose position is to be verified.
        pos : ndarray
            The hypothetical position of the electrode to verify.
        other_key : str
            The key of another electrode, used in updating non-overlapping dictionary. Default is None
        other_pos : ndarray
            The position of the other electrode. Default is None
        copy_of_non_overlapping_dict : dict
            A dictionary containing positions of electrodes without overlap, to be updated.

        Returns
        -------
        float
            A similarity score representing how well the `pos` aligns with the neighborhood
            defined in `icp_aligned`. Higher scores indicate better alignment. The score is
            zero if there are no common neighbors.

        Notes
        -----
        The similarity score is based on two factors:
        - Distance similarity: How close the distance to each neighbor is compared to the ICP data.
        - Direction similarity: How closely the unit direction vector aligns with the ICP data.
        """
        icp_neigh = find_electrode_neighbors(electrode_dict=icp_aligned, key=key)
        if other_pos is not None and other_key is not None:
            copy_of_non_overlapping_dict[other_key] = other_pos

        total_similarity = 0
        count = 0
        for k in icp_neigh:
            if k in copy_of_non_overlapping_dict:
                # Calculate the distance from the neighbor to the current point, and the unit vector
                dist_to_point = np.linalg.norm(pos - copy_of_non_overlapping_dict[k])

                # If dist to point is 0, skip this electrode
                if dist_to_point != 0:
                    unit_vector = (pos - copy_of_non_overlapping_dict[k]) / dist_to_point
                else:
                    continue

                diff_in_dist = 1 - np.abs(icp_neigh[k]['dist'] - dist_to_point)
                diff_in_dir = 1 - cosine(u=icp_neigh[k]['unit_direction'], v=unit_vector)

                total_similarity += (diff_in_dist + diff_in_dir) / 2
                count += 1

        # Overall similarity (average of individual similarities)
        if count > 0:
            return total_similarity / count
        else:
            return 0

    def _get_points_for_overlap_estimation(self, key, overlap, no_overlap, icp_temp):
        """
        Estimate the position of an electrode based on neighbor information.

        This function estimates the position of an electrode identified by 'key'.
        It uses the positions of verified neighboring electrodes and provides two estimates:
        one considering only the neighbors and the other considering all available data.

        Parameters:
        key: str
            The key identifying the electrode whose position is to be estimated.
        overlap: dict
            A dictionary containing current overlapping electrode positions.
        no_overlap: dict
            A dictionary containing positions of electrodes without overlap.
        icp_temp: dict
            A dictionary with ICP (Iterative Closest Point) aligned electrode data.

        Returns:
            tuple:
                A tuple containing the current position from 'overlap', and two estimated positions.
        """
        # Find neighbors of the electrode within a specified radius and filter out unverified ones
        neighbor_dict = find_electrode_neighbors(electrode_dict=icp_temp, radius=0.075, key=key)
        verified_neighbors = {n: neighbor_dict[n] for n in neighbor_dict if n in no_overlap}

        # Estimate electrode position considering only verified neighbors
        est = self._estimate_electrode(verified_electrodes=no_overlap, verified_neighbors=verified_neighbors,
                                       only_neighbors=True)

        # Estimate electrode position considering all available data
        est2 = self._estimate_electrode(verified_electrodes=no_overlap, verified_neighbors=verified_neighbors,
                                        only_neighbors=False)

        # Return the current overlapping position and the two estimates
        return overlap[key], est, est2

    def _handle_overlapping_and_unverified_electrodes(self, combined_discrep, verified_rgb_with_overlap, icp_rgb,
                                                      disc_rgb, verified_hsv_with_overlap, icp_hsv, disc_hsv,
                                                      verified_no_overlap_rgb, verified_no_overlap_hsv):
        """ Function to handle the overlapping electrodes.

        1. Loop through the overlapping electrode pair
        2. If the pair does not exist in RGB but in HSV, use the positions from RGB and vice versa
        3. If the discrepancy exist in both, use the function _handle_overlap
        4. Append the found files to the dictionary final_electrodes and the estimated to the estimated dictionary
        5. Return final and estimated dict

        Parameters
        ----------
        combined_discrep: list
            electrode pair overlapping
        verified_rgb_with_overlap: dict
            key, pos pair with all electrodes including the overlapping electrodes
        icp_rgb: dict
            icp aligned dictionary
        disc_rgb: list
            electrode pair overlapping in RGB
        verified_hsv_with_overlap: dict
            key, pos pair with all electrodes including the overlapping electrodes
        icp_hsv: dict
            icp aligned dictionary
        disc_hsv: list
            electrode pair overlapping in RGB
        verified_no_overlap_rgb: dict
            key, pos pair with all electrodes excluding the overlapping electrodes
        verified_no_overlap_hsv: dict
            key, pos pair with all electrodes excluding the overlapping electrodes

        Returns
        -------

        Notes:
        -------
        - There is an edge case in this function which is not handled here. If an electrode is a part of a discrepancy
        pair, in both RGB and HSV but not within the same pair. This edge case can be investigating by plotting the
        overlapping set in the bottom of this function. But a more robust estimate can be achieved by using the
        function final_verification with all electrodes, and is therefore not handled in this function!

        """

        def is_discrepancy_in_hsv_only(elec_par):
            """ Nested function to check whether the overlap is only in the HSV dictionary, returns True if that is the
            case.
            """
            return elec_par not in disc_rgb and elec_par in disc_hsv

        def is_discrepancy_in_rgb_only(elec_par):
            """ Nested function to check whether the overlap is only in the RGB dictionary, returns True if that is the
            case.
            """
            return elec_par not in disc_hsv and elec_par in disc_rgb

        def is_discrepancy_in_both(elec_par):
            """ Nested function to check if the overlap is in both rgb and hsv. Returns True if that is the case.
            """
            return elec_par in disc_hsv and elec_par in disc_rgb

        def update_electrodes_from_rgb(elec_par):
            """ Updates the final_electrode dict if the electrode in the pair is a part of the non-overlapping dict and
            not already a part of the final_electrode dict
            """
            for idx in range(2):
                if elec_par[idx] in verified_no_overlap_rgb and elec_par[idx] not in final_electrodes:
                    final_electrodes[elec_par[idx]] = verified_no_overlap_rgb[elec_par[idx]]
                if elec_par[idx] not in verified_no_overlap_rgb:
                    not_in_rgb.append(elec_par[idx])

        def update_electrodes_from_hsv(elec_par):
            """ Updates the final_electrode dict if the electrode in the pair is a part of the non-overlapping dict and
            not already a part of the final_electrode dict
            """
            for idx in range(2):
                if elec_par[idx] in verified_no_overlap_hsv and elec_par[idx] not in final_electrodes:
                    final_electrodes[elec_par[idx]] = verified_no_overlap_hsv[elec_par[idx]]
                if elec_par[idx] not in verified_no_overlap_hsv:
                    not_in_hsv.append(elec_par[idx])

        def update_electrodes_from_both(elec_par):
            """ Updates the final_electrode dict by calling function handle_overlap. Then the key and pos of the
            estimated electrode is added to the estimated dict.
            """
            e1_pos, e2_pos, est = self._handle_overlap(electrode_key1=elec_par[0], electrode_key2=elec_par[1],
                                                       overlap_rgb=verified_rgb_with_overlap,
                                                       no_overlap_rgb=verified_no_overlap_rgb, icp_rgb=icp_rgb,
                                                       overlap_hsv=verified_hsv_with_overlap, icp_hsv=icp_hsv,
                                                       no_overlap_hsv=verified_no_overlap_hsv)
            if est == 1:
                estimated[disc_par[1]] = e2_pos
            else:
                estimated[disc_par[0]] = e1_pos

            final_electrodes[disc_par[0]] = e1_pos
            final_electrodes[disc_par[1]] = e2_pos

        disc_hsv = list(disc_hsv)
        disc_rgb = list(disc_rgb)
        not_in_rgb, not_in_hsv = [], []
        final_electrodes = {}
        estimated = {}
        for disc_par in combined_discrep:
            # Check if the overlapping pair is only ion the hsv solution, if so, use the non-overlapping from RGB
            if is_discrepancy_in_hsv_only(elec_par=disc_par):
                update_electrodes_from_rgb(elec_par=disc_par)
            # Check if the overlapping pair is only ion the rgb solution, if so, use the non-overlapping from hsv
            elif is_discrepancy_in_rgb_only(elec_par=disc_par):
                update_electrodes_from_hsv(elec_par=disc_par)
            # Check if the overlapping pair is within both rgb and hsv
            elif is_discrepancy_in_both(elec_par=disc_par):
                update_electrodes_from_both(elec_par=disc_par)
            else:
                raise ValueError('This should not happen, as the missing keys, should be in either one of them....')

        # Edge case electrodes which will be handled in the final verification function!
        overlapping = set(not_in_rgb).intersection(set(not_in_hsv))

        return final_electrodes, estimated

    def final_verification(self, final_electrodes, icp_aligned, plot_and_print=False):
        """
        Perform final verification and adjustment of electrode positions.

        This method checks the positions of electrodes in the 'final_electrodes' dictionary and adjusts them
        if necessary to improve their alignment with neighboring electrodes based on the 'icp_aligned' data. It can
        also estimate positions for missing electrodes.

        Parameters:
        -----------
        final_electrodes: dict
            A dictionary containing electrode positions where keys are electrode labels.
        icp_aligned: dict
            Data representing the alignment information, possibly containing neighbor information.
        plot_and_print: bool
            If True, the method will print and plot the results for debugging.

        Returns:
        ---------
        - final_electrodes (dict): The updated dictionary of electrode positions.
        - estimated_electrodes (dict): A dictionary containing estimated electrode positions.

        """
        neighbor_dict = find_electrode_neighbors(electrode_dict=icp_aligned, radius=0.15)

        estimated_electrodes = {}
        for key in self._template_dict:
            if key == "Nz":
                continue

            if key in final_electrodes:
                # Check the current position against it neighbors
                original_check = self._verify_with_neighbors(icp_aligned=icp_aligned, key=key,
                                                             pos=final_electrodes[key],
                                                             copy_of_non_overlapping_dict=final_electrodes.copy())

                # If the error is below
                if original_check < 0.95:
                    verified_neighbors = {n_key: neighbor_dict[key][n_key]
                                          for n_key in neighbor_dict[key]
                                          if n_key in final_electrodes}
                    # Estimate the position and verify the new position against it neighbours
                    new_position = self._estimate_electrode(verified_electrodes=final_electrodes,
                                                            verified_neighbors=verified_neighbors)
                    new_check = self._verify_with_neighbors(icp_aligned=icp_aligned, key=key,
                                                            pos=new_position,
                                                            copy_of_non_overlapping_dict=final_electrodes.copy())
                    # Calculate the error and reduction
                    original_error = 1 - original_check
                    new_error = 1 - new_check
                    error_reduction = original_error - new_error
                    error_reduction_percentage = (error_reduction / original_error) * 100

                    # If error_reduction is above 10 percent
                    if error_reduction_percentage >= 10:  # Threshold for improvement
                        if plot_and_print:
                            print(key, original_check, new_check)

                            print(
                                f"{key}: Original Error: {original_error}, New Error: {new_error}, "
                                f"Reduction: {error_reduction_percentage}%")
                            self._plot_mesh(rotation_matrix=self._rotation_matrix,
                                            points=[final_electrodes[key], new_position],
                                            labels=['Org', 'New'])
                            print(f"Updated {key} with improved position.")

                        final_electrodes[key] = new_position
                        estimated_electrodes[key] = new_position
            else:
                # Estimate the electrode
                verified_neighbors = {n_key: neighbor_dict[key][n_key]
                                      for n_key in neighbor_dict[key]
                                      if n_key in final_electrodes}
                position = self._estimate_electrode(verified_electrodes=final_electrodes,
                                                    verified_neighbors=verified_neighbors)

                if plot_and_print:
                    print(f"ESTIMATING: {key}")
                    self._plot_mesh(rotation_matrix=self._rotation_matrix, points=[position], labels=[key])

                # Update both of the dictionaries
                final_electrodes[key] = position
                estimated_electrodes[key] = position

        return final_electrodes, estimated_electrodes

    def localize_electrodes(self):
        """ This function produces the finished solution of labeling of the electrodes.
        """
        # Align and label the two found arrays with RGB and HSV with the template using the verified midline
        elect_rgb, elect_hsv, icp_rgb, icp_hsv = (
            self._align_template_and_label(verified_electrodes=self._verified_midline.copy(),
                                           midline=True, plot_results=False))
        # Verify, label, align the HSV and RGB
        verified_rgb_with_overlap, estimated_rgb, disc_rgb, verified_without_overlap_rgb, icp_rgb = (
            self.label_align_fix_solution(found_electrodes=elect_rgb, icp_dictionary=icp_rgb))
        verified_hsv_with_overlap, estimated_hsv, disc_hsv, verified_without_overlap_hsv, icp_hsv = (
            self.label_align_fix_solution(found_electrodes=elect_hsv, icp_dictionary=icp_hsv))

        # 1. Merge the verified electrodes
        final_solution = self._merge_verified_electrodes(verified_rgb=verified_without_overlap_rgb,
                                                         verified_hsv=verified_without_overlap_hsv,
                                                         use_weighting=True)
        combined_discrepancies = disc_hsv.union(disc_rgb)

        # 2. Look at the overlapping electrodes?
        new_electrodes, estimated_overlap = (
            self._handle_overlapping_and_unverified_electrodes(combined_discrep=combined_discrepancies,
                                                               verified_rgb_with_overlap=verified_rgb_with_overlap,
                                                               icp_rgb=icp_rgb, disc_rgb=disc_rgb,
                                                               verified_hsv_with_overlap=verified_hsv_with_overlap,
                                                               icp_hsv=icp_hsv, disc_hsv=disc_hsv,
                                                               verified_no_overlap_rgb=verified_without_overlap_rgb,
                                                               verified_no_overlap_hsv=verified_without_overlap_hsv))

        # Update the dictionary with new values
        final_solution = update_dictionary(old_dict=final_solution, new_dict=new_electrodes)

        # 3. Final check of the solution, that all electrodes are present, the icp could have used rgb as well
        icp_avg = merge_dicts(dict1=icp_rgb, dict2=icp_hsv)
        finished_solution, estimated = self.final_verification(final_electrodes=final_solution,
                                                               icp_aligned=icp_avg)

        # 4. Set the found values
        self._final_estimates = set(
            list(estimated_rgb) + list(estimated_hsv) + list(estimated_overlap) + list(estimated))
        self._finished_solution = finished_solution

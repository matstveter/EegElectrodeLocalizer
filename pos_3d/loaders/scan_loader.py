import csv
import os
import time

import numpy as np

from pos_3d.classes.landmarks import LandmarksClass
from pos_3d.utils.automatic_verification import compare_fpz, read_fpz_locations
from pos_3d.verification.electrode_verification import ElectrodeVerifier
from pos_3d.verification.landmarkverifier import LandmarkVerifier
from pos_3d.classes.orientationanalyzer import MeshOrientation
from pos_3d.labelers.labeler import Labeler
from pos_3d.localizers.color_localization import ColorLocalizer
from pos_3d.labelers.midlinelabeler import MidlineLabeler
from pos_3d.localizers.fpz_color import FPzFinder
from pos_3d.utils.helper_functions import create_subject_folder, get_mesh_files, read_elc_file

print_timing = True


def read_csv_into_list():
    filename = "/home/tvetern/PhD/localize_electrodes/finished_subjects.csv"
    subject_ids = []
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                subject_id = row[0]
                subject_ids.append(subject_id)
        return subject_ids
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []


def write_subject_id_to_csv(subject_id, filename="approved"):
    if filename == "approved":
        filename = "/home/tvetern/PhD/localize_electrodes/approved_subjects.csv"
    else:
        filename = "/home/tvetern/PhD/localize_electrodes/finished_subjects.csv"
    try:
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([subject_id])
        print(f"Subject ID {subject_id} added to csv file successfully.")
    except Exception as e:
        print(f"Error writing to {filename}: {e}")


def print_times(start, end, str_information, path):
    file_path = f"{path}/timing.txt"
    with open(file_path, 'a') as file:
        if print_timing:
            info_width = 30  # Adjust this width as needed for your formatting
            info_str = f"{str_information:<{info_width}}"
            file.write(f"{info_str} : {end - start} seconds\n")


def get_all_scans(dataset_path, template_path, output_path, config):
    subject_folders = os.listdir(dataset_path)
    subject_folders.sort()
    template_dict, label_array, pos_array = read_elc_file(file_path=template_path, scale=1000)

    landmark_distances = []

    marked_subjects = []
    for i, folders in enumerate(subject_folders):
        subject_time = time.time()
        #################################################################################################
        # GET SUBJECT DATA, CREATE A SUBJECT FOLDER FOR RESULTS
        #################################################################################################
        start_time = time.time()
        subject_id = folders.split(sep=".")[0]
        subject_id = folders[:9]

        print(f"Current Subject ID: {subject_id}")

        subject_output_folder, logger = create_subject_folder(output_path=output_path,
                                                              subject_id=subject_id)
        print_times(start=start_time, end=time.time(), str_information="Loading Data", path=subject_output_folder)

        jpg_file, obj_file = get_mesh_files(dataset_path=dataset_path, subject_folder=folders,
                                            texture_file_format="jpg", mesh_format="obj")

        kwargs = {'obj_file': obj_file, 'jpg_file': jpg_file, 'subject_id': subject_id, 'logger': logger}

        #################################################################################################
        # DETECT ORIENTATION OF THE SCAN AND ALIGN IT SO THAT THE TOP OF THE HEAD IS TOWARDS Z-AXIS
        #################################################################################################
        start_time = time.time()
        orient = MeshOrientation(**kwargs)
        orient.detect_orientation()
        print_times(start=start_time, end=time.time(), str_information="Orientation Detection",
                    path=subject_output_folder)

        # Add important variables to the kwargs dictionary, used in subsequent classes
        kwargs['orientation_dictionary'] = orient.point_dict
        kwargs['rotation_matrix'] = orient.detected_rotation_matrix

        if not config['set_fiducials_manually']:
            #################################################################################################
            # FIND THE LANDMARKS - NASION, LPA, RPA AND ESTIMATE INION
            #################################################################################################
            start_time = time.time()
            landmarks = LandmarksClass(**kwargs)
            landmarks.get_landmarks()
            print_times(start=start_time, end=time.time(), str_information="Landmark Detection",
                        path=subject_output_folder)
            #################################################################################################
            # VERIFY THE LANDMARKS
            #################################################################################################
            start_time = time.time()
            landmark_verifier = LandmarkVerifier(nasion=landmarks.nasion, inion=landmarks.inion,
                                                 lpa=landmarks.lpa,
                                                 rpa=landmarks.rpa, origo=landmarks.origo,
                                                 target_folder=subject_output_folder, true_positions=None,
                                                 **kwargs)
            print_times(start=start_time, end=time.time(), str_information="Landmark Verification",
                        path=subject_output_folder)

            if config['use_detected_fiducials']:
                path = os.path.join(config['path_detected_fiducials'], subject_id)
                if not os.path.exists(path):
                    raise ValueError(f"Path: {path} for using already detected fiducials does not exist!")
                dists = landmark_verifier.get_found_landmarks(path=path)
                landmark_distances.append(dists)
                fpz_verified = read_fpz_locations(path=path)

                if config['verify_fiducials']:
                    landmark_verifier.manual_pick()

            else:
                if (config['manual_adjustment_if_red_flag'] and landmarks.red_flag) or config['verify_fiducials']:
                    landmark_verifier.manual_pick()
        else:
            val = np.array([0, 0, 0])
            landmark_verifier = LandmarkVerifier(nasion=val, inion=val, lpa=val, rpa=val, origo=val,
                                                 target_folder=subject_output_folder, true_positions=None, **kwargs)
            landmark_verifier.manual_pick()

        # Save the landmark positions
        landmark_verifier.write_to_csv(estimated=False)

        # add to kwargs array as these are used in all classes.
        kwargs["nasion"] = landmark_verifier.nasion
        kwargs["inion"] = landmark_verifier.inion
        kwargs["origo"] = landmark_verifier.origo
        kwargs["lpa"] = landmark_verifier.lpa
        kwargs["rpa"] = landmark_verifier.rpa

        #################################################################################################
        # FPZ FINDER - USED IN COMBINATION WITH NASION TO LATER DO TEMPLATE ALIGNMENT 
        #################################################################################################
        start_time = time.time()
        fpz_class = FPzFinder(template_dict=template_dict, **kwargs)
        fpz_class.localize_electrodes()
        print_times(start=start_time, end=time.time(), str_information="Fpz Detection",
                    path=subject_output_folder)

        if fpz_class.red_flag:
            logger.error("Can not find FPz!")
            marked_subjects.append(subject_id)
            if config["manual_adjustment_if_red_flag"]:
                print("Pick Fpz electrode, or the one that is just above Nasion.")
                fpz_class.pick_fpz(return_point=False)
            else:
                print("This is currently implemented as an important step, "
                      "so suggests to set manual_adjustment_if_red_flag to True")
                print("Manually select the Fpz electrode, or the one just above the Nasion if Fpz is not available")
                fpz_class.pick_fpz(return_point=False)

        #################################################################################################
        # SEGMENTATION OF THE MESH ACCORDING TO COLORS - THIS WILL GIVE BOTH A RGB AND A HSV POINT CLOUD 
        #################################################################################################
        start_time = time.time()
        col = ColorLocalizer(**kwargs)
        col.localize_electrodes()
        print_times(start=start_time, end=time.time(), str_information="Electrode Detection",
                    path=subject_output_folder)

        if col.red_flag:
            if col.suggested_cluster_centers_rgb == 0 and col.suggested_cluster_centers_hsv == 0:
                logger.error("No electrodes found, skipping rest of method")
                marked_subjects.append(subject_id)
                continue

        #################################################################################################
        # LABEL AND ESTIMATE THE MIDLINE ACROSS THE CENTRE IN BOTH DIRECTION 
        #################################################################################################
        start_time = time.time()
        mlab = MidlineLabeler(template_dict=template_dict, fpz=fpz_class.fpz,
                              suggested_electrode_positions_rgb=col.suggested_cluster_centers_rgb,
                              suggested_electrode_positions_hsv=col.suggested_cluster_centers_hsv, **kwargs)
        mlab.localize_electrodes()
        print_times(start=start_time, end=time.time(), str_information="Midline Labeler",
                    path=subject_output_folder)

        #################################################################################################
        # NEW TEMPLATE ALIGNMENT BASED ON THE FOUND MIDLINE, LABEL/ESTIMATE THE REST OF THE MISSING  
        #################################################################################################
        start_time = time.time()
        # Use the found_hsv and found_rgb from the midline-labeler, as the estimated points is added here
        lab = Labeler(template_dict=template_dict,
                      verified_midline=mlab.midline,
                      found_hsv=mlab.found_hsv,
                      found_rgb=mlab.found_rgb, **kwargs)
        lab.localize_electrodes()
        print_times(start=start_time, end=time.time(), str_information="Labeler",
                    path=subject_output_folder)

        #################################################################################################
        # VERIFICATION OF ALL ELECTRODES, WITH THE FUNCTIONALITY OF ADJUSTING POINTS 
        #################################################################################################
        if config['use_detected_fiducials']:
            ret = compare_fpz(fpz_verified=fpz_verified, fpz_found=lab.finished_solution['FPz'])
            ver = ElectrodeVerifier(final_electrodes=lab.finished_solution,
                                    estimated_electrodes=lab.final_estimates,
                                    target_folder=subject_output_folder,
                                    **kwargs)
            ver.verifying_positions(verify=config['visual_validation'])
            if ret:
                write_subject_id_to_csv(subject_id=subject_id, filename="approved")
            else:
                print("FPz too far away---")
                logger.error("Fpz too far away...")
        else:

            ver = ElectrodeVerifier(final_electrodes=lab.finished_solution,
                                    estimated_electrodes=lab.final_estimates,
                                    target_folder=subject_output_folder,
                                    **kwargs)
            ver.verifying_positions(verify=config['visual_validation'])

            print_times(start=subject_time, end=time.time(), str_information="Subject Finished",
                        path=subject_output_folder)

            write_subject_id_to_csv(subject_id=subject_id)

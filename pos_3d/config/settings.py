import configparser
import os


def get_config_settings():
    """ Returns settings specified in pos_3d/config/3d_config.ini

    This function uses the configparser function to extract information from the config file.

    Returns
    -------
    dict: Dictionary containing settings, such as paths, and so on

    Raises:
    FileNotFoundError: If the config file does not exist

    """
    config = configparser.ConfigParser()
    # Extract the path to the folder, and then merge this with the name of the config file
    ret_conf = config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), '3d_config.ini'))

    if not ret_conf:
        raise FileNotFoundError('Could not find the specified config file in the folder pos_3d/config/...\nSpecify'
                                'correct path in the pos_3d/config/setting.py, line 17')

    current_working_directory = os.getcwd()

    setting_dict = dict()
    try:
        setting_dict["dataset_path"] = config.get('PATH', 'dataset_path')
        setting_dict["template_path"] = os.path.join(current_working_directory,
                                                     config.get('PATH', 'template_path'))
        setting_dict["output_path"] = os.path.join(current_working_directory,
                                                   config.get('PATH', 'output_path'))

        setting_dict["set_fiducials_manually"] = config.getboolean('FIDUCIALS', 'set_fiducials_manually',
                                                                   fallback=True)
        setting_dict["verify_landmarks"] = config.getboolean('FIDUCIALS', 'verify_landmarks',
                                                             fallback=True)

        setting_dict["visual_validation"] = config.getboolean('ELECTRODES', 'visual_validation',
                                                              fallback=False)
        setting_dict["manual_adjustment_if_red_flag"] = config.getboolean('GENERAL', 'manual_adjustment_if_red_flag',
                                                                          fallback=False)

        setting_dict['use_detected_fiducials'] = config.getboolean('OTHER', 'use_detected_fiducials')
        setting_dict['path_detected_fiducials'] = config.get('OTHER', 'path_detected_fiducials')
    except configparser.NoOptionError as e:
        print(f"Missing option in configuration file: {e}")
        raise
    except configparser.NoSectionError as e:
        print(f"Missing section in configuration file: {e}")
        raise

    if setting_dict["set_fiducials_manually"] and setting_dict['use_detected_fiducials']:
        raise ValueError("[ERROR] Both set_fiducials_manually and use_detected_fiducials are set to True, "
                         "which makes little "
                         "sense.")

    if not setting_dict['visual_validation']:
        print("[WARNING] The visual validation flag in the config file is set to False. Meaning that the electrode"
              "solution will no be shown, and does not allow for manual adjustments. Set to True if this is wanted.")

    return setting_dict

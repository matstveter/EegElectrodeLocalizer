import os
import sys

from pos_3d.config.settings import get_config_settings
from pos_3d.loaders.scan_loader import get_all_scans
import pyvista as pv

from pos_3d.utils.helper_functions import create_run_folder


def run_project():
    # Get information from the config file
    config = get_config_settings()
    output_path = create_run_folder(path=config['output_path'])
    get_all_scans(dataset_path=config['dataset_path'], template_path=config['template_path'],
                  output_path=output_path, config=config)

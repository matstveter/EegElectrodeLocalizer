import csv
import multiprocessing
import os
import tkinter as tk
import numpy as np
import pyvista as pv

from pos_3d.localizers.base_localizer import BaseLocalizer
from pos_3d.utils.mesh_helper_functions import manual_point_selection


class ElectrodeVerifier(BaseLocalizer):

    def __init__(self, rotation_matrix, orientation_dictionary, nasion, inion, lpa, rpa, origo,
                 final_electrodes, estimated_electrodes, target_folder, **kwargs):
        super().__init__(rotation_matrix=rotation_matrix, orientation_dictionary=orientation_dictionary,
                         nasion=nasion, inion=inion, lpa=lpa, rpa=rpa, origo=origo, **kwargs)

        self._final_electrodes = final_electrodes
        self._estimated_electrodes = estimated_electrodes

        self._target_folder = target_folder

        self.queue = multiprocessing.Queue()
        self.close_event = multiprocessing.Event()  # Event to signal PyVista process closure

    def _mesh_solution_plotter(self):

        plotter = pv.Plotter()

        # Read the mesh and texture using the pyvista library
        mesh = pv.read(self._obj_file)
        texture = pv.read_texture(self._jpg_file)

        # Transform the mesh according to the rotation matrix
        mesh.points = np.dot(mesh.points, self._rotation_matrix.T)
        mesh.points = mesh.points - self._point_transformation

        plotter.add_mesh(mesh=mesh, texture=texture)

        for key, val in self._final_electrodes.items():
            sphere = pv.Sphere(radius=0.002, center=val)
            if key in self._estimated_electrodes:
                plotter.add_mesh(sphere, color="y")
            else:
                plotter.add_mesh(sphere, color="g")

        plotter.add_point_labels(points=list(self._final_electrodes.values()),
                                 labels=list(self._final_electrodes.keys()),
                                 font_size=15,
                                 always_visible=True,
                                 shape_color="w")

        plotter.show(window_size=[5000, 5000])
        plotter.close()

    def write_to_csv(self, dict_to_write, estimated=True):
        """
        Write electrode positions to a CSV file.

        Parameters:
        -----------
        dict_to_write : dict
            A dictionary where keys represent electrode names, and values are lists [x, y, z] representing their
            positions.
        estimated : bool, optional
            Indicates whether the positions are estimated (default) or verified.

        Returns:
        --------
        None
        """
        if estimated:
            fields = ['Electrode', 'x', 'y', 'z', 'was_estimated']
            with open(os.path.join(self._target_folder, f"estimated_electrode_positions.csv"), mode="w",
                      newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # Write the header row
                csvwriter.writerow(fields)
                # Write each row of data
                for electrode, values in dict_to_write.items():
                    if electrode == "Nz":
                        continue

                    if electrode in self._estimated_electrodes:
                        was_estimated = 1
                    else:
                        was_estimated = 0

                    row = [electrode, f"{values[0]:.6f}", f"{values[1]:.6f}", f"{values[2]:.6f}", f"{was_estimated}"]
                    csvwriter.writerow(row)
        else:
            fields = ['Electrode', 'x', 'y', 'z']
            with open(os.path.join(self._target_folder, f"verified_electrode_positions.csv"), mode="w",
                      newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # Write the header row
                csvwriter.writerow(fields)
                # Write each row of data
                for electrode, values in dict_to_write.items():
                    if electrode == "Nz":
                        continue
                    row = [electrode, f"{values[0]:.6f}", f"{values[1]:.6f}", f"{values[2]:.6f}"]
                    csvwriter.writerow(row)

    def create_button_window(self):
        window = tk.Tk()
        screen_number = 0
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        screen_x = window.winfo_vrootx() + screen_number * screen_width
        screen_y = window.winfo_vrooty()

        # Set the window's geometry to appear on the specified screen
        window.geometry(f"{screen_width}x{screen_height}+{screen_x}+{screen_y}")

        # Sort keys based on the first letter and length
        sorted_keys = sorted(self._final_electrodes.keys(), key=lambda x: (x[0], len(x), x))

        # Group keys by starting letter
        grouped_keys = {}
        for key in sorted_keys:
            starting_letter = key[0]
            if starting_letter not in grouped_keys:
                grouped_keys[starting_letter] = []
            grouped_keys[starting_letter].append(key)

        # Calculate maximum length per column
        max_length_per_column = 15

        # Create buttons and place them in the grid with overflow to the next column
        col = 0
        for starting_letter, keys in grouped_keys.items():
            if starting_letter == "N":
                continue
            row = 0
            for key in keys:
                button = tk.Button(window, text=key, font=10, width=6, height=2,
                                   command=lambda k=key: on_button_press(k))
                button.grid(row=row, column=col, padx=5, pady=5)
                row += 1

                # Move to the next column if the maximum length is reached
                if row == max_length_per_column:
                    row = 0
                    col += 1

            col += 1

        finished_button = tk.Button(window, text="Finished", font=25, width=40, height=2,
                                    command=lambda: on_button_press("Finito"))
        # Place the 'Finished' button at the bottom
        finished_button.grid(row=max_length_per_column + 1, columnspan=col, padx=10, pady=10)

        def on_button_press(but_key):
            self.queue.put(but_key)
            window.destroy()

        def on_window_close():
            self.queue.put("Finito")
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", on_window_close)

        window.mainloop()

    def verify(self):
        """ This function calls the creation of a button window with all the electrode keys as seperate buttons and a
        plotting function over the estimated electrode positions. If a button other than finished is pressed, a new
        function to update the position of that pressed button is called. Loops until the "Finished" button is pressed.

        Returns
        -------

        """
        while True:
            pyvista_process = multiprocessing.Process(target=self._mesh_solution_plotter)
            pyvista_process.start()

            button_process = multiprocessing.Process(target=self.create_button_window)
            button_process.start()

            pyvista_process.join()
            button_process.join()

            selected_key = self.queue.get()

            if selected_key == "Finito":
                break

            self._update_key(point_to_update=selected_key)

    def _update_key(self, point_to_update):

        picked_point = manual_point_selection(obj_file_path=self._obj_file,
                                              jpg_file_path=self._jpg_file,
                                              rotation_matrix=self._rotation_matrix,
                                              point_transform=self._point_transformation,
                                              point_to_be_picked=point_to_update,
                                              suggested_point=self._final_electrodes[point_to_update])

        if picked_point is not None:
            # Update the final_verification arr
            self._final_electrodes[point_to_update] = picked_point

    def verifying_positions(self, verify=True):
        self.write_to_csv(dict_to_write=self._final_electrodes, estimated=True)
        if verify:
            self.verify()
            self.write_to_csv(dict_to_write=self._final_electrodes, estimated=False)

    def localize_electrodes(self):
        """ Not needed for this class"""
        pass

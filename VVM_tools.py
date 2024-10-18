import os
import re
import numpy as np
from netCDF4 import Dataset

class VVM_tools:
    def __init__(self, directory):
        """
        Initializes the data loader by scanning a directory of NetCDF files and finding variable-to-file mappings.
        
        :param directory: Path to the folder containing NetCDF files.
        """
        self.directory = directory
        self.files = self._scan_files()
        self.var_file_map = self._find_variable_file_relationship()

    def _parse_filename(self, filename):
        """
        Parse the filename into case name, file type, and time.
        :param filename: The NetCDF filename (e.g., pbl_ctl.C.Surface-000235.nc)
        :return: (case_name, file_type, time)
        """
        match = re.match(r"^([^.]+)\.([^.]+(?:\.[^.]+)*)-(\d+)\.nc$", filename)
        if match:
            case_name, file_type, time_str = match.groups()
            return case_name, file_type, time_str
        else:
            raise ValueError(f"Filename '{filename}' does not match the expected pattern.")
    
    def _scan_files(self):
        """
        Scans the directory and organizes files based on case, file type, and time.
        :return: A dictionary with structure {case_name: {file_type: {time: file_path}}}
        """
        file_dict = {}
        
        for file in os.listdir(self.directory):
            if file.endswith(".nc"):
                try:
                    case_name, file_type, time_str = self._parse_filename(file)
                    
                    # Organize the files in a nested dictionary
                    if case_name not in file_dict:
                        file_dict[case_name] = {}
                    if file_type not in file_dict[case_name]:
                        file_dict[case_name][file_type] = {}
                    
                    # Store the file path based on the time
                    file_dict[case_name][file_type][time_str] = os.path.join(self.directory, file)
                except ValueError as e:
                    print(f"Skipping file '{file}': {e}")
        
        return file_dict

    def _find_variable_file_relationship(self):
        """
        Reads the first available file for each file type (e.g., t=0 files) to find the relationship between variables and file types.
        :return: A dictionary where keys are variable names and values are the file types that contain the variables.
        """
        var_file_map = {}

        # Loop through each file type and extract the variables from a t=0 file
        for case_name, file_types in self.files.items():
            for file_type, time_files in file_types.items():
                # Get any file for t=0 or the first time step
                t0_file = next(iter(time_files.values()))
                
                with Dataset(t0_file, 'r') as nc_file:
                    variables = list(nc_file.variables.keys())
                    for var in variables:
                        # Assign the file type to each variable
                        var_file_map[var] = file_type
        
        return var_file_map

    def get_variable(self, case_name, variable_name, time):
        """
        Finds the appropriate file and extracts the variable's data, allowing time to be input as an integer.
        
        :param case_name: Case name of the experiment (e.g., 'pbl_ctl').
        :param variable_name: Name of the variable to extract.
        :param time: Time step as either an integer or a string (e.g., 235 or '000235').
        :return: A numpy array containing the variable's data.
        """
        # Convert integer time to zero-padded string if needed
        if isinstance(time, int):
            time = f"{time:06d}"  # Convert to a 6-character zero-padded string
        elif isinstance(time, str) and len(time) != 6:
            raise ValueError("Time string must be exactly 6 characters long.")
        
        # Find the file type based on the variable name
        var_type = self.var_file_map.get(variable_name)
        if var_type is None:
            raise ValueError(f"Variable '{variable_name}' not found in any file type.")
        
        try:
            file_path = self.files[case_name][var_type][time]
            with Dataset(file_path, 'r') as nc_file:
                # Extract the variable as a numpy array
                variable_data = np.array(nc_file.variables[variable_name][:])
            return variable_data
        except KeyError:
            raise FileNotFoundError(f"File for case '{case_name}', variable type '{var_type}', and time '{time}' not found.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the variable '{variable_name}': {e}")

    def get_variable_file_type(self, variable_name):
        """
        Returns the file type (e.g., C.Surface, L.Dynamic) that contains the specified variable.
        
        :param variable_name: The name of the variable whose file type is to be found.
        :return: The file type as a string.
        """
        file_type = self.var_file_map.get(variable_name)
        if file_type is None:
            raise ValueError(f"Variable '{variable_name}' not found in any file type.")
        return file_type

    def get_all_variables_and_file_types(self):
        if not self.var_file_map:
            print("No variables found. Please ensure the files are properly loaded.")
            return

        # Create a mapping from file types to their variables
        type_var_map = {}
        for var_name, file_type in self.var_file_map.items():
            if file_type not in type_var_map:
                type_var_map[file_type] = []
            type_var_map[file_type].append(var_name)

        # Print the data types and their corresponding variables
        print("Data Type -> Variables")
        print("=========================")
        for file_type, variables in type_var_map.items():
            print(f"{file_type}: {', '.join(sorted(variables))}")

if __name__ == "__main__":
    loader = VVM_tools("/data/chung0823/VVM_cloud_dynamics_2024/DATA/pbl_ctl/archive/")
    loader.get_all_variables_and_file_types()
    
    # Example: Get a variable data using integer time
    data = loader.get_variable("pbl_ctl", "th", 235) 
    print(data)
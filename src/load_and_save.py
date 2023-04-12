import pandas as pd
import argparse
from src.utils.common_utils import log, clean_prev_dirs_if_exis, create_dir, save_raw_local_df

def get_data(raw_data_path, log_file):
    """
        get the data from Raw Data folder & store to artifacts directory:\n
        :param raw_data_path: raw_data_path
        :param directory_path: folder_name
        :param new_data_path: new_data_path
        :return: data
    """
    try:
        data = pd.read_csv(raw_data_path, sep=',') # read the data
        file = log_file
        log(file_object=file, log_message=f"read the data, shape: {data.shape}")  # logs the details
        return data


    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the error if occurs
        raise e

def save_data(data, directory_path, new_data_path, log_file):
    """
        It helps to save the data.
        :param data: data
        :param directory_path: directory_path
        :param new_data_path: new_data_path
        :return: save data
    """
    try:
        file = log_file
        data = data
        # clean_prev_dirs_if_exist(dir_path=directory_path) # cleaned directory if exists
        # create_dir(dirs=[directory_path]) # create directory.
        # log(file_object=file, log_message=f"create directory for storing the data: {directory_path}")  # logs the details

        save_raw_local_df(data=data, data_path=new_data_path)
        log(file_object=file, log_message=f"store data in : {new_data_path}")  # logs the details
        return data  # return data

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the error if occurs
        raise e

if __name__ == "__main__":
    pass



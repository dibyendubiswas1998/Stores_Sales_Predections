from src.utils.common_utils import  log, clean_prev_dirs_if_exis, create_dir, save_raw_local_df
from sklearn.model_selection import train_test_split


def split_and_save_data(data, log_file, directory_path, train_data_path, evaluation_data_path, split_ratio, random_state):
    """
    It helps to split & save the data.\n
    :param data: data
    :param log_file: log_file.txt
    :param directory_path: directory_path
    :param train_data_path: train_data_path
    :param evaluation_data_path: evaluation_data_path
    :param split_ratio: split_ratio
    :param random_state: random_state
    :return: train & evaluation
    """
    try:
        data = data
        file = log_file

        train, evaluation = train_test_split(data, test_size=split_ratio, random_state=random_state) # split data in train & evaluation set
        log(file_object=log_file, log_message=f"split ghe data in tran.csv: {train.shape} & evaluation.csv: {evaluation.shape}") # logs the details

        clean_prev_dirs_if_exis(dir_path=directory_path)  # cleaned directory if exists
        create_dir(dirs=[directory_path])  # create directory.
        log(file_object=file,
            log_message=f"create directory for storing the data: {directory_path}")  # logs the details

        for data, data_path in (train, train_data_path), (evaluation, evaluation_data_path):
            save_raw_local_df(data, data_path)
            log(file_object=file, log_message=f"store data in : {data_path}")  # logs the details

        log(file_object=file, log_message=f"Splitting Operation completed")  # logs the details
        return train, evaluation  # return train, evaluation dataset

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the error if occurs
        raise e


if __name__ == '__main__':
    pass

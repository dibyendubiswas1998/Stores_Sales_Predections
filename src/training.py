from src.load_and_save import get_data, save_data
from src.split_and_save_ import split_and_save_data
from src.feature_engineering import mean_encoding, target_encoding
from src.utils.common_utils import read_params, log, clean_prev_dirs_if_exis, create_dir, save_raw_local_df, save_report
from src.model_creation import model_creation
from src.evaluation import model_evaluation
import json
from datetime import datetime


def training(config_path:str):
    """
        It helps to train the model:
        :param config_path: params.yaml
        :return: none
    """
    try:
        now = datetime.now()
        date = now.date()
        current_time = now.strftime("%H:%M:%S")

        config = read_params(config_path=config_path) # read params.yaml file
        training_logs_file = config['artifacts']['log_files']['training_log_file'] # artifacts/Logs/training_logs.txt
        log(file_object=training_logs_file, log_message="training process starts")


        # Step-01: load & save train.csv data
        log(file_object=training_logs_file, log_message="Step-1: load & save train.csv data")
        raw_train_data_path = config['data_source']['train_path'] # get Raw Data/Train_new.csv
        train = get_data(raw_data_path=raw_train_data_path, log_file=training_logs_file) # get the train data.

        artifacts = config['artifacts'] # artifacts
        raw_data_dir = artifacts['raw_data']['raw_data_dir'] # get artifacts/Raw_Data directory
        new_train_path = artifacts['raw_data']['train_path'] # get artifacts/Raw_Data/train.csv path
        save_data(data=train, directory_path=raw_data_dir, new_data_path=new_train_path, log_file=training_logs_file) # save the data to artifacts/Raw_Data directory
        log(file_object=training_logs_file, log_message="successfully data is loaded and save\n\n")


        # Step-02: split the train.csv --> train.csv & evaluation.csv and store in a directory.
        log(file_object=training_logs_file, log_message="Step-2: Start to split the data")
        random_state = config['base']['random_state']  # get random state
        split_ratio = config['base']['split_ratio']  # get split ratio

        split_data_dir = artifacts['split_data']['split_data_dir'] # artifacts/Split_Data directory
        split_train_path = artifacts['split_data']['train_path'] # artifacts/Split_Data/train.csv file path
        split_evaluation_path  = artifacts['split_data']['evaluation_path'] # artifacts/Split_Data/evaluation.csv file path
        train = get_data(raw_data_path=new_train_path, log_file=training_logs_file) # load artifacts/Raw_Data/train.csv
        train, evaluation = split_and_save_data(data=train,
                                                log_file=training_logs_file,
                                                directory_path=split_data_dir,
                                                train_data_path=split_train_path,
                                                evaluation_data_path=split_evaluation_path,
                                                split_ratio=split_ratio,
                                                random_state=random_state) # split the data into train.csv & evaluation.csv
        log(file_object=training_logs_file, log_message="Successfully split the data\n\n")


        # Step-03: Apply Encoding technique (mean encoding & target encoding) & store the data:
        log(file_object=training_logs_file, log_message="Step-3: Apply Encoding technique (mean encoding & target encoding) & store the data")
        ycol = config['data_defination']['output_col'] # ycol: output_col
        xcol = config['data_defination']['xcols'] # xcols

        metrics_dir = artifacts['matrices']['metrics_dir'] # artifacts/Matrices directory
        metrics_file_path = artifacts['matrices']['metrics_file_path'] # artifacts/Matrices/key_matrix.json file

        clean_prev_dirs_if_exis(dir_path=metrics_dir) # remove artifacts/Matrices directory if present
        create_dir(dirs=[metrics_dir]) # create artifacts/Matrices directory

        for col in xcol:
            train, train_dct = mean_encoding(data=train, xcol=col, ycol=ycol, log_file=training_logs_file) # apply mean encoding in train.csv
            key_matrix = {
                col: train_dct,
            } # store dict items for each column.
            save_report(file_path=metrics_file_path, report=key_matrix) # save the key_matrix in artifacts/Matrices directory
            evaluation, evaluation_dct = mean_encoding(data=train, xcol=col, ycol=ycol, log_file=training_logs_file) # apply mean encoding evaluation.csv

        processed_dir = artifacts['processed_data']['processed_dir'] # artifacts/Processed_Data
        pro_train_path = artifacts['processed_data']['train_path'] # artifacts/Processed_Data/train.csv
        pro_evaluation_path = artifacts['processed_data']['evaluation_path'] # artifacts/Processed_Data/evaluation.csv

        clean_prev_dirs_if_exis(dir_path=processed_dir) # remove artifacts/Processed_Data directory if present
        create_dir(dirs=[processed_dir]) # create artifacts/Processed_Data directory

        # store processed train.csv & evaluation.csv data to Processed_Data directory
        for data, data_path in (train, pro_train_path), (evaluation, pro_evaluation_path):
            save_raw_local_df(data, data_path)
        log(file_object=training_logs_file, log_message=f"store data in : {pro_train_path} & {pro_evaluation_path}")  # logs the details
        log(file_object=training_logs_file, log_message="Successfully perform Mean Encoding technique\n\n")


        # Step-04: Create model based on artifacts/Processed_Data/train.csv data.
        log(file_object=training_logs_file, log_message="Step-4: Start to create the model")
        train_path_ = artifacts['processed_data']['train_path'] # artifacts/Processed_Data/train.csv
        train_ = get_data(raw_data_path=train_path_, log_file=training_logs_file) # load data from artifacts/Processed_Data/train.csv

        model_dir = artifacts['model']['model_dir'] # artifacts/Model directory
        model_path = artifacts['model']['model_path'] # artifacts/Model/model.joblib file

        model_creation(train_data=train_, ycol=ycol,
                       model_directory=model_dir, model_path=model_path,
                       log_file=training_logs_file) # create model
        log(file_object=training_logs_file, log_message="Successfully Model is created\n\n")


        # Step-05: Evaluate the model performance & save score in the score.json:
        log(file_object=training_logs_file, log_message="Step-5: Start to Evaluate the Model Performance")
        model_path_ = artifacts['model']['model_path']  # artifacts/Model/model.joblib file
        reports_dir = artifacts['report']['reports_dir'] # artifacts/Model_Performance_Report directory
        score_file_path = artifacts['report']['scores'] # artifacts/Model_Performance_Report/score.json file

        train_path__ = artifacts['processed_data']['train_path'] # artifacts/Processed_Data/train.csv
        evaluation_path__ = artifacts['processed_data']['evaluation_path'] # artifacts/Processed_Data/evaluation.csv

        model_evaluation(train_path=train_path__,
                         evaluation_data_path=evaluation_path__,
                         ycol=ycol,
                         model_path=model_path_,
                         report_dir=reports_dir,
                         score_file_path=score_file_path,
                         log_file=training_logs_file) # start to evaluate the model and store the score.
        log(file_object=training_logs_file, log_message="Successfully Store the Model Performance report")
        return "model training successfully completed"


    except Exception as e:
        print(e)
        config = read_params(config_path=config_path)  # read params.yaml file
        training_logs_file = config['artifacts']['log_files']['training_log_file']  # artifacts/Logs/training_logs.txt
        log(file_object=training_logs_file, log_message=f"Error will be {e} \n\n")
        raise e





import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from src.utils.common_utils import read_params, log, create_dir, clean_prev_dirs_if_exis, save_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json


def eval_matrics(actual_val, predt_val):
    """
        It helps to get different evaluation matrices.
        :param actual_val: actual value
        :param predt_val: predicted value
        :return: rmse, mse, r2
    """
    mae = mean_absolute_error(actual_val, predt_val)
    mse = mean_squared_error(actual_val, predt_val)
    r2 = r2_score(actual_val, predt_val)
    rmse = np.sqrt(((predt_val - actual_val) ** 2).mean())
    return mae, mse, r2, rmse


def model_evaluation(train_path, evaluation_data_path, ycol, model_path, report_dir, score_file_path, log_file):
    """
        It helps to evaluate model performance.\n
        :param train_path: train_path
        :param evaluation_data_path: evaluation_data_path
        :param ycol: ycol
        :param model_path: model_path
        :param report_dir: report_dir
        :param report_dir: report_dir.json
        :param log_file: log_file
        :return: none
    """
    try:
        evaluation = pd.read_csv(evaluation_data_path) # read evaluation.csv data
        train = pd.read_csv(train_path) # read train.csv data
        file = log_file

        evaluation_y = evaluation[ycol] # get the output_col  evaluation data
        evaluation_x = evaluation.drop(columns=[ycol], axis=1) # drop the output_col: item_outlet_sales

        train_y = train[ycol] # get the output_col  from train data


        scale = StandardScaler() # applying standardization on test.csv data
        evaluation_x = scale.fit_transform(evaluation_x) # scaled the test data.
        log(file_object=file, log_message="applying standardization on evaluation data")  # logs the details

        # load the model from Model directory:
        lr = joblib.load(model_path) # load the model.
        log(file_object=log_file, log_message=f"load the model for predictions from {model_path}")  # logs the details

        predicted_val = lr.predict(evaluation_x) # predict the score.
        mae, mse, r2, rmse = eval_matrics(evaluation_y, predicted_val)

        clean_prev_dirs_if_exis(dir_path=report_dir) # remove if previously present.
        create_dir(dirs=[report_dir]) # create Model_Performance_Report directory.
        log(file_object=file, log_message=f"create directory for storing the scores, path: {report_dir}")  # logs the details

        scores = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2,
            "train_range_op": [min(train_y), max(train_y)],
            "evaluation_range_op": [min(evaluation_y), max(evaluation_y)]
        }
        save_report(score_file_path, scores)
        log(file_object=file, log_message=f"save the report & score will be: {scores}")  # logs the details



    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the error if occurs
        raise e


if __name__ == '__main__':
    pass


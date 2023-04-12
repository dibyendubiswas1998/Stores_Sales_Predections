import pandas as pd
import argparse
from src.utils.common_utils import read_params, log, clean_prev_dirs_if_exis, create_dir, save_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib


def model_creation(train_data, ycol, model_directory, model_path, log_file):
    """
        It helps to create the model based on train data set.\n
        :param train_data: train.csv
        :param ycol: ycol
        :param model_directory: model_directory
        :param model_path: model_path
        :param log_file: log_file.txt
        :return: none
        """
    try:
        data = train_data
        file = log_file
        log(file_object=file, log_message="\n\nmodel Creation process is start") # logs the details

        y_train = data[ycol] # get the output feature
        x_train = data.drop(columns=[ycol], axis=1) # get the features data
        log(file_object=file, log_message="separate x_train & y_train  feature") # logs the details

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train) # scaled the data using StandardScaler() method
        log(file_object=file, log_message="scaled the x_train data using StandardScaler() method")  # logs the details


        # Applying GridSearchCV:
        # grid_param = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
        #          'C': [1, 5, 10],
        #          'degree': [3, 8, 13],
        #          'coef0': [0.01, 10, 0.5],
        #          'gamma': ('auto', 'scale')
        #          },
        # svr_ = SVR() # applying SVR algorithm initially
        # grid_search = GridSearchCV(estimator=svr_,
        #                            param_grid=grid_param,
        #                            cv=5,
        #                            n_jobs=-1)
        # grid_search.fit(x_train, y_train) # apply GridSearchCV()
        # best_parameters = grid_search.best_params_ # read the best parameter after hyperparameter tuning.
        # kernel = best_parameters['kernel']
        # C = best_parameters['C']
        # degree = best_parameters['degree']
        # coef0 = best_parameters['coef0']
        # gamma = best_parameters['gamma']
        # log(file_object=log_file, log_message=f"getting the parameter after applying GridSearchCV(), params: {best_parameters}") # logs the details
        #
        # # apply SVR algorithm, after hyperparameter tuning.
        # model = SVR(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, C=C)
        model = RandomForestRegressor()
        model.fit(x_train, y_train)
        log(file_object=file, log_message="create the RandomForestRegressor model after getting params.") # logs the details

        # crating model directory inside artifacts' directory:
        clean_prev_dirs_if_exis(dir_path=model_directory)
        create_dir(dirs=[model_directory])
        save_model(model_name=model, model_path=model_path)
        log(file_object=file,
            log_message=f"save the model in {model_path} directory.")  # logs the details


    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e} \n\n")  # logs the error if occurs
        raise e



if __name__ == "__main__":
    pass


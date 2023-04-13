import joblib
from src.load_and_save import get_data, save_data
from src.split_and_save_ import split_and_save_data
from src.feature_engineering import mean_encoding, target_encoding
from src.utils.common_utils import read_params, log, clean_prev_dirs_if_exis, create_dir, save_raw_local_df, save_report
from src.model_creation import model_creation
from src.evaluation import model_evaluation
from sklearn.preprocessing import StandardScaler



def prediction(config_path):
    """
        It helps to predict the output based on test data.\n
        :param config_path: config_path
        :return: data
    """
    try:
        config = read_params(config_path=config_path)  # read params.yaml file
        prediction_logs_file = config['artifacts']['log_files']['prediction_log_file']  # artifacts/Logs/prediction_logs.txt
        log(file_object=prediction_logs_file, log_message="prediction process starts based on test data")

        # Step-01: load & save train.csv data
        log(file_object=prediction_logs_file, log_message="Step-1: load & save test.csv data")
        raw_test_data_path = config['data_source']['test_path']  # get Raw Data/Test_new.csv
        test = get_data(raw_data_path=raw_test_data_path, log_file=prediction_logs_file)  # get the test data.

        artifacts = config['artifacts']  # artifacts
        raw_data_dir = artifacts['raw_data']['raw_data_dir']  # get artifacts/Raw_Data directory
        new_test_path = artifacts['raw_data']['test_path']  # get artifacts/Raw_Data/test.csv path
        save_data(data=test, directory_path=raw_data_dir, new_data_path=new_test_path,
                  log_file=prediction_logs_file)  # save the data to artifacts/Raw_Data directory
        log(file_object=prediction_logs_file, log_message="successfully data is loaded and save\n\n")


        # Step-02: Apply Encoding technique (mean encoding & target encoding) & store the data:
        log(file_object=prediction_logs_file, log_message="apply the encoding method on test.csv dataset")
        test_path = artifacts['raw_data']['test_path'] # artifacts/Raw_Data/test.csv
        test = get_data(raw_data_path=test_path, log_file=prediction_logs_file) # get the test.csv data from artifacts/Raw_Data directory

        # metrics_file_path = config['artifacts']['matrices']['metrics_file_path'] # load the artifacts/Matrices/key_matrix.txt file
        item_fat_content_dct = {"regular": 2232.8780279127213, "low fat": 2185.4785638186377} # item_fat_content feature
        item_type_dct = {"starchy foods": 2458.4508953125, "seafood": 2367.2380291666664, "fruits and vegetables": 2321.843276846307,
                     "snack foods": 2295.1017390319257, "household": 2261.7029123655916, "canned": 2249.6992643274853,
                     "dairy": 2216.495402238806, "breakfast": 2210.9437372093025, "breads": 2205.4099368421053,
                     "hard drinks": 2203.0266907103824, "meat": 2191.2563766153844, "frozen foods": 2128.1382869309837,
                     "health and hygiene": 2048.9498498777507, "soft drinks": 2031.1150587570621,
                     "baking goods": 1980.0015603143418, "others": 1934.838749640288
                    } # item_type feature
        outlet_type_dct = {"supermarket": 2470.1089498573588, "grocery store": 344.99058696158323} # outlet_type feature
        outlet_location_type_dct = {"tier 2": 2333.5835031710585, "tier 3": 2306.68713081761, "tier 1": 1895.446358315565} # outlet_location_type feature
        outlet_size_mode_dct = {"high": 2347.469406989247, "medium": 2301.333706477927, "small": 1929.3022135362014} # # feature

        ycol = config['data_defination']['output_col']  # ycol: output_col
        xcol = config['data_defination']['xcols']  # xcols

        # applying target encoding based on dictionary
        test = target_encoding(data=test, xcol="item_fat_content", dct_map=item_fat_content_dct, log_file=prediction_logs_file) # apply target encoding on item_fat_content
        test = target_encoding(data=test, xcol="item_type", dct_map=item_type_dct, log_file=prediction_logs_file) # apply target encoding on item_type
        test = target_encoding(data=test, xcol="outlet_type", dct_map=outlet_type_dct, log_file=prediction_logs_file) # apply target encoding on outlet_type
        test = target_encoding(data=test, xcol="outlet_location_type", dct_map=outlet_location_type_dct, log_file=prediction_logs_file) # apply target encoding on outlet_location_type
        test = target_encoding(data=test, xcol="outlet_size_mode", dct_map=outlet_size_mode_dct, log_file=prediction_logs_file) # apply target encoding on outlet_size_mode

        # store data in Processed_Data directory.
        pro_test_path = artifacts['processed_data']['test_path']  # artifacts/Processed_Data/test.csv
        save_raw_local_df(data=test, data_path=pro_test_path) # save data to artifacts/Processed_Data
        log(file_object=prediction_logs_file, log_message="successfully applied the encoding method and store the data\n\n")


        # Step-3: Prediction.
        log(file_object=prediction_logs_file, log_message="start to predict the output value based on test.csv data")
        test_path = artifacts['processed_data']['test_path']  # artifacts/Processed_Data/test.csv
        test_ = get_data(raw_data_path=test_path, log_file=prediction_logs_file) # load the data from artifacts/Processed_Data

        # standardize the data:
        scaler = StandardScaler()  # StandardScaler method.
        test_scaled = scaler.fit_transform(test_)  # standardize the test.csv data.

        # load the model:
        model_path = artifacts['model']['model_path'] # load the model: artifacts/Model/model.joblib
        model = joblib.load(model_path) # load the model
        predicted_value = model.predict(test_scaled) # predict the value.

        prediction_dir = artifacts['prediction']['prediction_dir'] # artifacts/Prediction
        prediction_file_path = artifacts['prediction']['prediction_file'] # artifacts/Prediction/predict.csv
        clean_prev_dirs_if_exis(dir_path=prediction_dir) # remove directory if previously exist
        create_dir(dirs=[prediction_dir]) # create directory
        log(file_object=prediction_logs_file, log_message=f"create directory for storing the predicted data, path: {prediction_file_path}")

        output_col = config['data_defination']['output_col'] # mention output column.
        test_[output_col] = predicted_value # store in dataframe
        save_raw_local_df(data=test_, data_path=prediction_file_path) # store the predicted data in artifacts/Prediction directory
        log(file_object=prediction_logs_file, log_message="successfully predict the value & store data as a predict.csv format\n\n")

    except Exception as e:
        print(e)
        raise e

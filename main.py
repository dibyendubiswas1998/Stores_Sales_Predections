from src.utils.common_utils import read_params
from src.training import training
from src.prediction import prediction
import joblib
from flask import Flask, render_template, redirect, url_for, request


app = Flask(__name__)

@app.route("/")
def Home():
    return render_template('index.html')

@app.route("/prediction")
def Prediction():
    return render_template('prediction.html')

@app.route("/train")
def Train():
    result = training("params.yaml")
    print(result)
    return render_template("index.html", result=result)

@app.route("/test_prediction")
def Test_prediction():
    result = prediction("params.yaml")
    print(result)
    return render_template("index.html", result=result)

@app.route("/prediction")
def Start_Prediction():
    return redirect("prediction.html")

@app.route("/submit",  methods=['POST', 'GET'])
def Calculation():
    if request.method == 'POST':
        item_fat_content = request.form['item_fat_content']
        item_visibility = request.form['item_visibility']
        item_type = request.form['item_type']
        item_mrp = request.form['item_mrp']
        outlet_location_type = request.form['outlet_location_type']
        outlet_type = request.form['outlet_type']
        item_weight_mean = request.form['item_weight_mean']
        outlet_size = request.form['outlet_size']

        # load the artifacts/Matrices/key_matrix.txt file
        item_fat_content_dct = {"regular": 2232.8780279127213, "low fat": 2185.4785638186377}  # item_fat_content feature
        item_type_dct = {"starchy foods": 2458.4508953125, "seafood": 2367.2380291666664, "fruits and vegetables": 2321.843276846307,
                         "snack foods": 2295.1017390319257, "household": 2261.7029123655916, "canned": 2249.6992643274853,
                         "dairy": 2216.495402238806, "breakfast": 2210.9437372093025, "breads": 2205.4099368421053,
                         "hard drinks": 2203.0266907103824, "meat": 2191.2563766153844, "frozen foods": 2128.1382869309837,
                         "health and hygiene": 2048.9498498777507, "soft drinks": 2031.1150587570621,
                         "baking goods": 1980.0015603143418, "others": 1934.838749640288
                         }  # item_type feature
        outlet_type_dct = {"supermarket": 2470.1089498573588, "grocery store": 344.99058696158323}  # outlet_type feature
        outlet_location_type_dct = {"tier 2": 2333.5835031710585, "tier 3": 2306.68713081761, "tier 1": 1895.446358315565}  # outlet_location_type feature
        outlet_size_mode_dct = {"high": 2347.469406989247, "medium": 2301.333706477927, "small": 1929.3022135362014}  # # feature

        # get the value from dict:
        item_fat_content_val = item_fat_content_dct.get(item_fat_content)
        item_visibility_val = item_visibility
        item_type_val = item_type_dct.get(item_type)
        item_mrp_val = item_mrp
        outlet_location_type_val = outlet_location_type_dct.get(outlet_location_type)
        outlet_type_val = outlet_type_dct.get(outlet_type)
        item_weight_mean_val = item_weight_mean
        outlet_size_val = outlet_size_mode_dct.get(outlet_size)

        # new test data:
        new_test_data = [[item_fat_content_val, item_visibility_val, item_type_val,
                                           item_mrp_val, outlet_location_type_val, outlet_type_val,
                                           item_weight_mean_val, outlet_size_val]]

        # load the model:
        config = read_params("params.yaml")
        model_path = config['artifacts']['model']['model_path'] # get the path: artifacts/Model/model.joblib
        model = joblib.load(model_path)  # load the model
        result = model.predict(new_test_data)
        print(result)

        return f"store sales wii be: {result}"



if __name__ == "__main__":
    app.run(debug=True)


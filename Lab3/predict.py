import pickle
from flask import Flask, request, jsonify
import pandas as pd
import mlflow
import os

# load saved model

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
RUN_ID='-d6aaa9c14d0b434084f2dd9daaa5e697/artifacts/MLmodel'
RUN_ID = os.getenv('RUN_ID')

logged_model = f'mlflow-artifacts:/2/models/m{RUN_ID}/artifacts/model.pkl'
# logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)




def prepare_features(ride):
    return pd.DataFrame([{
        'trip_distance': ride['trip_distance'],
        'trip_duration': ride['trip_duration']
    }])

def predict(features_df):
    preds = model.predict(features_df)
    return float(preds[0])

app = Flask('fare-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features_df = prepare_features(ride)
    pred = predict(features_df)

    result = {
        'predicted_fare': pred
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

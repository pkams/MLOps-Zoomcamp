import pickle
from flask import Flask, request, jsonify
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

RUN_ID = 'b5eb4a60a4934bba8ccd86bf71d3f9fa'

def load_model(run_id):
    # Loading the make pipeline!
    logged_model = f'runs:/{RUN_ID}/models_pickle'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' %(ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    model = load_model(RUN_ID)
    preds = model.predict(features)
    return float(preds[0])

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(ride)

    result = {'duration': pred}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    # Using waitress
    # waitress-serve --port=port script:app_flask
    # waitress-serve --port=9696 predict:app

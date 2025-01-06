from flask import Flask, render_template, request, jsonify, Blueprint
import pandas as pd
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing, load_iris
from model.random_forest_regressor_model import RandomForestRegressionModel
from waitress import serve
from updated import scaled_features, features, data, corr_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)



# Load and train Random Forest Regression model
random_forest_model = RandomForestRegressionModel()
url = "https://data.nasa.gov/resource/e6wj-e2uc.json"
X_train, X_val, X_test, y_train, y_val, y_test = random_forest_model.load_and_preprocess_data(url)
random_forest_model.train(X_train, y_train)
model, scaler, scaled_features, feature_columns = joblib.load('final_model1.pkl')


# Blueprint for Random Forest Regression routes
random_forest_regression_bp = Blueprint('random_forest_regression', __name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')

@random_forest_regression_bp.route('/')
def random_forest_regression_home():
    return render_template('random_forest_regression.html')

@random_forest_regression_bp.route('/predict', methods=['POST'])
def predict_random_forest_regression():
    try:
        data = request.get_json(force=True)
        input_data = {feature: 0 for feature in feature_columns}  # Initialize all features to 0
        input_data.update({feature: data['features'][i] for i, feature in enumerate(feature_columns[:len(data['features'])])})
        features_df = pd.DataFrame([input_data])
        features_df[scaled_features] = scaler.transform(features_df[scaled_features])
        prediction = model.predict(features_df)

        # Evaluation metrics on the test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5  # Manually calculate RMSE
        r2 = r2_score(y_test, y_pred)

        # Plotting actual vs predicted values
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linestyle='--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values for Random Forest Regressor')

        img_path = 'static/plots/random_forest_regressor.png'
        plt.savefig(img_path)
        plt.close()  # Close the plot to avoid displaying

        return jsonify({
            'prediction': prediction[0],
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'plot_url': f'/{img_path}'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@random_forest_regression_bp.route('/graphs', methods=['GET'])
def get_graphs():
    try:
        # Generate and save graphs
        graphs = []

        # Histogram
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            img_path = f'static/plots/hist_{feature}.png'
            plt.savefig(img_path)
            plt.close()  # Close the plot to avoid displaying
            graphs.append(img_path)

        # Pairplot
        pairplot = sns.pairplot(data, vars=features)
        img_path = 'static/plots/pairplot.png'
        pairplot.savefig(img_path)
        plt.close()  # Close the plot to avoid displaying
        graphs.append(img_path)

        # Correlation Matrix
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        img_path = 'static/plots/correlation_matrix.png'
        plt.savefig(img_path)
        plt.close()  # Close the plot to avoid displaying
        graphs.append(img_path)

        return jsonify({'graphs': graphs})
    except Exception as e:
        return jsonify({'error': str(e)})


app.register_blueprint(random_forest_regression_bp, url_prefix='/random_forest_regression')



if __name__ == '__main__':
    serve(app, host='127.0.0.1', port=8080)

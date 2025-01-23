from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

app = Flask(__name__)

# Global variables to store the model and transformer
model = None
transformer = None
data = None

@app.route('/upload', methods=['POST'])
def upload_data():
    global data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        data = pd.read_csv(file)
        return jsonify({'message': 'Data uploaded successfully', 'shape': data.shape}), 200
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/train', methods=['POST'])
def train_model():
    global model, transformer, data
    if data is None:
        return jsonify({'error': 'No data uploaded'}), 400

    # Splitting features and target
    X = data.drop("Downtime", axis=1)
    y = data["Downtime"]

    # Date preprocessing
    X['Date'] = pd.to_datetime(X['Date'], format='%d-%m-%Y')
    X['day'] = X['Date'].dt.day_name()
    X['month'] = X['Date'].dt.month_name()
    X['year'] = X['Date'].dt.year
    X.drop(columns=['Date'], inplace=True)

    # Defining numerical and categorical columns
    num = [
        'year', 'Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)',
        'Air_System_Pressure(bar)', 'Coolant_Temperature',
        'Hydraulic_Oil_Temperature(?C)', 'Spindle_Bearing_Temperature(?C)',
        'Spindle_Vibration(?m)', 'Tool_Vibration(?m)', 'Spindle_Speed(RPM)',
        'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)'
    ]
    cat = ['Machine_ID', 'Assembly_Line_No']

    # Preprocessing pipelines
    num_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ordinal_day = Pipeline([
        ("mode", SimpleImputer(strategy="most_frequent")),
        ("ordinal_day", OrdinalEncoder(categories=[['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                                                    'Thursday', 'Friday', 'Saturday']],
                                       unknown_value=-1, handle_unknown="use_encoded_value"))
    ])

    ordinal_month = Pipeline([
        ("mode", SimpleImputer(strategy="most_frequent")),
        ("ordinal_month", OrdinalEncoder(categories=[['January', 'February', 'March', 'April', 'May',
                                                      'June', 'July', 'August', 'September',
                                                      'October', 'November', 'December']],
                                         unknown_value=-1, handle_unknown="use_encoded_value"))
    ])

    # Column transformer
    transformer = ColumnTransformer([
        ("numeric", num_preprocessor, num),
        ("categorical", cat_preprocessor, cat),
        ("ordinal_day", ordinal_day, ['day']),
        ("ordinal_month", ordinal_month, ['month'])
    ], remainder="passthrough").set_output(transform="pandas")

    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transforming data
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    # Training the model
    model = LogisticRegression()
    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_test_transformed)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Save the model and transformer
    joblib.dump(model, 'classifier.joblib')
    joblib.dump(transformer, 'transformer.joblib')

    return jsonify({
        'message': 'Model trained successfully',
        'accuracy': accuracy,
        'f1_score': f1
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    global model, transformer
    if model is None or transformer is None:
        return jsonify({'error': 'Model not trained'}), 400

    try:
        # Parse input JSON
        input_data = request.get_json(force=True)

        # Validate input data
        required_fields = [
            "Date", "Machine_ID", "Assembly_Line_No",
            "Hydraulic_Pressure(bar)", "Coolant_Pressure(bar)", "Air_System_Pressure(bar)",
            "Coolant_Temperature", "Hydraulic_Oil_Temperature(?C)", 
            "Spindle_Bearing_Temperature(?C)", "Spindle_Vibration(?m)",
            "Tool_Vibration(?m)", "Spindle_Speed(RPM)", "Voltage(volts)",
            "Torque(Nm)", "Cutting(kN)"
        ]
        for field in required_fields:
            if field not in input_data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Prepare input DataFrame
        input_df = pd.DataFrame([input_data])

        # Date preprocessing
        input_df['Date'] = pd.to_datetime(input_df['Date'], format='%d-%m-%Y')
        input_df['day'] = input_df['Date'].dt.day_name()
        input_df['month'] = input_df['Date'].dt.month_name()
        input_df['year'] = input_df['Date'].dt.year
        input_df.drop(columns=['Date'], inplace=True)

        # Transform input
        input_transformed = transformer.transform(input_df)

        # Make predictions
        prediction = model.predict(input_transformed)
        probability = model.predict_proba(input_transformed).max()
        # Return prediction
        print(prediction)
        print(probability)

        return jsonify({
            'Downtime': 'No_Machine_Failure' if prediction[0] == 1 else 'Machine_Failure',
            'Confidence': float(probability),
            'Prediction': prediction[0]
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

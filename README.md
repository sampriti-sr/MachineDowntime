# Machine Downtime Prediction API

This API predicts machine downtime based on input parameters using a trained logistic regression model. It includes endpoints for uploading data, training the model, and making predictions.

---

## Table of Contents
1. [Requirements](#requirements)
2. [Setup Instructions](#setup-instructions)
3. [Endpoints](#endpoints)
    - [/upload](#upload)
    - [/train](#train)
    - [/predict](#predict)
4. [Example Requests and Responses](#example-requests-and-responses)

---

## Requirements
Ensure you have the following installed on your system:
- Python 3.8+
- Flask
- pandas
- scikit-learn
- joblib

Install the required Python libraries using:
```bash
pip install flask pandas scikit-learn
```

---

## Setup Instructions
1. Clone the repository or download the script.
2. Save the script as `app.py`.
3. Run the Flask server:
   ```bash
   python app.py
   ```
4. The server will start at `http://127.0.0.1:5000/`.

---

## Endpoints

### 1. `/upload`
**Method**: `POST`
**Description**: Uploads a CSV file containing the training data.

#### Request
- **Headers**:
  - `Content-Type: multipart/form-data`
- **Body**:
  - File upload with key `file`. The file must be in `.csv` format.

#### Response
- **Success (200)**:
  ```json
  {
      "message": "Data uploaded successfully",
      "shape": [1000, 15]
  }
  ```
- **Failure (400)**:
  ```json
  {
      "error": "No file part"
  }
  ```

### 2. `/train`
**Method**: `POST`
**Description**: Trains a logistic regression model using the uploaded data.

#### Request
- **Headers**:
  - `Content-Type: application/json`

#### Response
- **Success (200)**:
  ```json
  {
      "message": "Model trained successfully",
      "accuracy": 0.92,
      "f1_score": 0.91
  }
  ```
- **Failure (400)**:
  ```json
  {
      "error": "No data uploaded"
  }
  ```

### 3. `/predict`
**Method**: `POST`
**Description**: Predicts machine downtime based on input parameters.

#### Request
- **Headers**:
  - `Content-Type: application/json`
- **Body**:
  ```json
  {
      "Date": "31-12-2021",
      "Machine_ID": "Makino-L1-Unit1-2013",
      "Assembly_Line_No": "Shopfloor-L1",
      "Hydraulic_Pressure(bar)": 71.04,
      "Coolant_Pressure(bar)": 6.9337,
      "Air_System_Pressure(bar)": 6.2849,
      "Coolant_Temperature": 25.6,
      "Hydraulic_Oil_Temperature(?C)": 46,
      "Spindle_Bearing_Temperature(?C)": 33.4,
      "Spindle_Vibration(?m)": 1.291,
      "Tool_Vibration(?m)": 26.492,
      "Spindle_Speed(RPM)": 25892,
      "Voltage(volts)": 335,
      "Torque(Nm)": 24.0553,
      "Cutting(kN)": 3.58
  }
  ```

#### Response
- **Success (200)**:
  ```json
  {
      "Downtime": "No",
      "Confidence": 0.95
  }
  ```
- **Failure (400)**:
  ```json
  {
      "error": "Missing field: Hydraulic_Pressure(bar)"
  }
  ```

---

## Example Requests and Responses

### 1. Upload Data
#### Request (RESTful API):
**POST** `http://127.0.0.1:5000/upload`

**Headers**:
```
Content-Type: multipart/form-data
```

**Body**:
- Form data:
  - Key: `file`
  - Value: `<your_csv_file>`

#### Response:
```json
{
    "message": "Data uploaded successfully",
    "shape": [2000, 16]
}
```

### 2. Train the Model
#### Request (RESTful API):
**POST** `http://127.0.0.1:5000/train`

**Headers**:
```
Content-Type: application/json
```

#### Response:
```json
{
    "message": "Model trained successfully",
    "accuracy": 0.92,
    "f1_score": 0.91
}
```

### 3. Make a Prediction
#### Request (RESTful API):
**POST** `http://127.0.0.1:5000/predict`

**Headers**:
```
Content-Type: application/json
```

**Body**:
```json
{
    "Date": "31-12-2021",
    "Machine_ID": "Makino-L1-Unit1-2013",
    "Assembly_Line_No": "Shopfloor-L1",
    "Hydraulic_Pressure(bar)": 71.04,
    "Coolant_Pressure(bar)": 6.9337,
    "Air_System_Pressure(bar)": 6.2849,
    "Coolant_Temperature": 25.6,
    "Hydraulic_Oil_Temperature(?C)": 46,
    "Spindle_Bearing_Temperature(?C)": 33.4,
    "Spindle_Vibration(?m)": 1.291,
    "Tool_Vibration(?m)": 26.492,
    "Spindle_Speed(RPM)": 25892,
    "Voltage(volts)": 335,
    "Torque(Nm)": 24.0553,
    "Cutting(kN)": 3.58
}
```

#### Response:
```json
{
    "Downtime": "No",
    "Confidence": 0.95
}
```

---

## Notes
- Ensure the uploaded dataset contains a `Downtime` column for training.
- Preprocess the input data correctly to match the expected format.
- For any issues, raise an error or check the logs.

---

## Contact
For questions or support, please contact [srsampriti@gmail.com].


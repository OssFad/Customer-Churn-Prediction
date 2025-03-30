from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel

# Load models, encoders, and scalers
with open("best_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler_data = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define request model
class InputData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Function to preprocess new input data
def preprocess_input(input_data: dict):
    input_df = pd.DataFrame([input_data])
    
    # Apply encoding to categorical features
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Apply scaling to numerical features
    for col, scaler in scaler_data.items():
        if col in input_df.columns:
            input_df[col] = scaler.transform(input_df[[col]])
    
    return input_df

# Function to make prediction
def make_prediction(input_data: dict):
    input_df = preprocess_input(input_data)
    prediction = loaded_model.predict(input_df)[0]
    probability = loaded_model.predict_proba(input_df)[0, 1]
    return {"prediction": "Churn" if prediction == 1 else "No Churn", "probability": probability}

# Define API endpoint
@app.post("/predict")
async def predict_churn(data: InputData):
    input_dict = data.dict()
    result = make_prediction(input_dict)
    return result

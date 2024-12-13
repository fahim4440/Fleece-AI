from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now; restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods, including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# Load your TensorFlow model
# model = tf.keras.models.load_model('saved_model/yarn_model.h5')
model = tf.keras.models.load_model('saved_model/h5_model.h5')
scaler = joblib.load('saved_model/scaler.pkl')

# Define the input schema
class YarnInput(BaseModel):
    gsm: float
    cotton: float
    polyester: float
    elastane: float

# API endpoint for predictions
@app.post("/predict/")
async def predict(data: YarnInput):
    # Prepare input for the model
    input_data = np.array([[data.gsm, data.cotton, data.polyester, data.elastane]])

    scaled_input = scaler.transform(input_data)

    # Perform predictions
    regression_output, classification_output = model.predict(scaled_input)

    # Decode classification outputs
    knit_type = np.argmax(classification_output[0][:3])
    binder_type = np.argmax(classification_output[0][3:6])
    loop_type = np.argmax(classification_output[0][6:])

    # Return results
    return {
        "regression": {
            "knit_yarn_count": float(regression_output[0][0]),
            "binder_yarn_count": float(regression_output[0][1]),
            "loop_yarn_count": float(regression_output[0][2]),
            "knit_sl": float(regression_output[0][3]),
            "binder_sl": float(regression_output[0][4]),
            "loop_sl": float(regression_output[0][5]),
        },
        "classification": {
            "knit_yarn_type": int(knit_type),
            "binder_yarn_type": int(binder_type),
            "loop_yarn_type": int(loop_type),
        }
    }

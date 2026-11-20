
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

# Create FastAPI application
app = FastAPI(
    title="Hotel Booking Prediction API",
    description="Predict hotel reservation status",
    version="1.0"
)

# Load saved ML model
model_path = Path(__file__).resolve().parent / "hotel_reservation_model.pkl"

model = joblib.load(model_path)

print("Hotel booking model loaded successfully!")

# Input features
class BookingData(BaseModel):

    hotel: str
    lead_time: int

    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int

    stays_in_weekend_nights: int
    stays_in_week_nights: int

    adults: int
    children: float
    babies: int

    meal: str
    country: str
    market_segment: str
    distribution_channel: str

    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int

    reserved_room_type: str
    deposit_type: str
    customer_type: str

    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int

    city: str


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to Hotel Booking Prediction API"
    }


# Prediction endpoint
@app.post("/predict")
def predict_booking(booking: BookingData):

    # Convert input data into dictionary
    booking_data = booking.dict()

    # Convert dictionary into DataFrame
    new_booking = pd.DataFrame([booking_data])

    # Predict reservation status
    prediction = model.predict(new_booking)

    # Get prediction probabilities
    probabilities = model.predict_proba(new_booking)[0]

    # Prepare probability results
    probability_result = {}

    for status, probability in zip(model.classes_, probabilities):

        probability_result[str(status)] = round(
            float(probability * 100), 2
        )

    return {
        "predicted_reservation_status": str(prediction[0]),
        "prediction_probabilities": probability_result
    }


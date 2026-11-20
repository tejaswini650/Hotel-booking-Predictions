
from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Load trained ML pipeline
model_path = Path(__file__).resolve().parent / "hotel_reservation_model.pkl"
model = joblib.load(model_path)

print("Hotel booking model loaded successfully!")

@app.route("/")
def home():

    booking_data = {
        "hotel": "City Hotel",
        "lead_time": 50,
        "arrival_date_year": 2026,
        "arrival_date_month": "July",
        "arrival_date_week_number": 28,
        "arrival_date_day_of_month": 10,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "IND",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "deposit_type": "No Deposit",
        "customer_type": "Transient",
        "adr": 100.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 1,
        "city": "Hyderabad"
    }

    return render_template("index.html", booking_data=booking_data)


@app.route("/predict", methods=["POST"])
def predict():

    booking_data = {
        "hotel": request.form["hotel"],
        "lead_time": int(request.form["lead_time"]),
        "arrival_date_year": int(request.form["arrival_date_year"]),
        "arrival_date_month": request.form["arrival_date_month"],
        "arrival_date_week_number": int(request.form["arrival_date_week_number"]),
        "arrival_date_day_of_month": int(request.form["arrival_date_day_of_month"]),
        "stays_in_weekend_nights": int(request.form["stays_in_weekend_nights"]),
        "stays_in_week_nights": int(request.form["stays_in_week_nights"]),
        "adults": int(request.form["adults"]),
        "children": float(request.form["children"]),
        "babies": int(request.form["babies"]),
        "meal": request.form["meal"],
        "country": request.form["country"],
        "market_segment": request.form["market_segment"],
        "distribution_channel": request.form["distribution_channel"],
        "is_repeated_guest": int(request.form["is_repeated_guest"]),
        "previous_cancellations": int(request.form["previous_cancellations"]),
        "previous_bookings_not_canceled": int(request.form["previous_bookings_not_canceled"]),
        "reserved_room_type": request.form["reserved_room_type"],
        "deposit_type": request.form["deposit_type"],
        "customer_type": request.form["customer_type"],
        "adr": float(request.form["adr"]),
        "required_car_parking_spaces": int(request.form["required_car_parking_spaces"]),
        "total_of_special_requests": int(request.form["total_of_special_requests"]),
        "city": request.form["city"]
    }

    new_booking = pd.DataFrame([booking_data])

    prediction = model.predict(new_booking)[0]

    probabilities = model.predict_proba(new_booking)[0]

    probability_result = {}

    for status, probability in zip(model.classes_, probabilities):
        probability_result[str(status)] = round(float(probability * 100), 2)

    return render_template(
        "index.html",
        booking_data=booking_data,
        prediction=prediction,
        probabilities=probability_result
    )


if __name__ == "__main__":
    app.run(debug=True)

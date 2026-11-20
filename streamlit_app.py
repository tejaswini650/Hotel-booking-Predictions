
import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Hotel Booking Prediction",
    page_icon="🏨", 
    layout="centered"
)

st.title("🏨 Hotel Booking Reservation Prediction")
st.write("Enter booking details to predict reservation status.")

# FastAPI URL
API_URL = "http://127.0.0.1:8000/predict"

# Booking form
with st.form("booking_form"):

    st.subheader("Hotel Details")

    hotel = st.selectbox(
        "Hotel",
        ["City Hotel", "Resort Hotel"]
    )

    lead_time = st.number_input("Lead Time", min_value=0, value=50)

    st.subheader("Arrival Details")

    arrival_date_year = st.number_input(
        "Arrival Year", min_value=2015, max_value=2030, value=2017
    )

    arrival_date_month = st.selectbox(
        "Arrival Month",
        ["January", "February", "March", "April",
         "May", "June", "July", "August",
         "September", "October", "November", "December"]
    )

    arrival_date_week_number = st.number_input(
        "Arrival Week Number", min_value=1, max_value=53, value=32
    )

    arrival_date_day_of_month = st.number_input(
        "Arrival Day", min_value=1, max_value=31, value=10
    )

    st.subheader("Stay Details")

    stays_in_weekend_nights = st.number_input(
        "Weekend Nights", min_value=0, value=2
    )

    stays_in_week_nights = st.number_input(
        "Week Nights", min_value=0, value=3
    )

    adults = st.number_input("Adults", min_value=0, value=2)
    children = st.number_input("Children", min_value=0.0, value=0.0)
    babies = st.number_input("Babies", min_value=0, value=0)

    st.subheader("Booking Information")

    meal = st.selectbox(
        "Meal",
        ["BB", "HB", "SC", "FB", "Undefined"]
    )

    country = st.text_input("Country Code", value="PRT")

    market_segment = st.selectbox(
        "Market Segment",
        ["Online TA", "Offline TA/TO", "Groups",
         "Direct", "Corporate", "Complementary", "Aviation",
         "Undefined"]
    )

    distribution_channel = st.selectbox(
        "Distribution Channel",
        ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
    )

    is_repeated_guest = st.selectbox(
        "Repeated Guest", [0, 1]
    )

    previous_cancellations = st.number_input(
        "Previous Cancellations", min_value=0, value=0
    )

    previous_bookings_not_canceled = st.number_input(
        "Previous Bookings Not Canceled", min_value=0, value=0
    )

    reserved_room_type = st.text_input(
        "Reserved Room Type", value="A"
    )

    deposit_type = st.selectbox(
        "Deposit Type",
        ["No Deposit", "Non Refund", "Refundable"]
    )

    customer_type = st.selectbox(
        "Customer Type",
        ["Transient", "Transient-Party", "Contract", "Group"]
    )

    st.subheader("Additional Details")

    adr = st.number_input("ADR", min_value=0.0, value=100.0)

    required_car_parking_spaces = st.number_input(
        "Required Parking Spaces", min_value=0, value=0
    )

    total_of_special_requests = st.number_input(
        "Total Special Requests", min_value=0, value=1
    )

    city = st.text_input("City", value="Hyderabad")

    submitted = st.form_submit_button("Predict Reservation Status")

# Send data to FastAPI
if submitted:

    booking_data = {
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country.strip().upper(),
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
        "city": city
    }

    try:
        response = requests.post(
            API_URL,
            json=booking_data,
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction completed successfully!")

            st.subheader("Predicted Reservation Status")

            st.title(result["predicted_reservation_status"])

            st.subheader("Prediction Probabilities")

            probabilities = result["prediction_probabilities"]

            for status, probability in probabilities.items():

                st.write(f"**{status}: {probability:.2f}%**")

                st.progress(
                    min(max(float(probability) / 100, 0.0), 1.0)
                )

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Please start the FastAPI server first."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"Request failed: {error}")

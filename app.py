# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import joblib

# # Initialize Flask app
# app = Flask(__name__)

# # Load the trained model and scaler
# model = joblib.load("disease_outbreak_model.pkl")
# scaler = joblib.load("scaler.pkl")

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get input data from the form
#         temperature = float(request.form["temperature"])
#         humidity = float(request.form["humidity"])
#         rainfall = float(request.form["rainfall"])
#         population = int(request.form["population"])
#         population_density = float(request.form["population_density"])
#         urbanization_rate = float(request.form["urbanization_rate"])
#         hospitals = int(request.form["hospitals"])
#         beds = int(request.form["beds"])
#         doctors = int(request.form["doctors"])
#         nurses = int(request.form["nurses"])

#         # Prepare the input data for prediction
#         input_data = np.array([[temperature, humidity, rainfall, population, population_density,
#                                 urbanization_rate, hospitals, beds, doctors, nurses]])
#         input_data_scaled = scaler.transform(input_data)

#         # Make the prediction
#         prediction = model.predict(input_data_scaled)[0]
#         probability = model.predict_proba(input_data_scaled)[0][1]  # Probability of high outbreak

#         # Generate the result
#         result = {
#             "prediction": "High Outbreak Risk" if prediction == 1 else "Low Outbreak Risk",
#             "confidence": f"{probability:.2%}" if prediction == 1 else f"{1 - probability:.2%}",
#         }
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)




# from flask import Flask, request, jsonify, render_template, redirect, url_for
# import pandas as pd
# import numpy as np
# import joblib
# import requests

# # Initialize Flask app
# app = Flask(__name__)

# # Load the trained model and scaler
# model = joblib.load("disease_outbreak_model.pkl")
# scaler = joblib.load("scaler.pkl")

# # Load data
# climate_data = pd.read_csv("climate_data_monthly.csv")  # Climate data
# population_data = pd.read_csv("population_data.csv")    # Population data
# healthcare_data = pd.read_csv("healthcare_data.csv")    # Healthcare data

# # OpenWeatherMap API configuration
# API_KEY = "2c84a58594dea5397b1a0b140de2ebcf"  # Replace with your actual API key
# BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# @app.route("/")
# def index():
#     """Render the main form page."""
#     return render_template("index.html")

# @app.route("/get_states")
# def get_states():
#     """Return a list of unique states."""
#     states = population_data["State"].unique().tolist()
#     return jsonify({"states": states})

# @app.route("/get_districts")
# def get_districts():
#     """Return a list of districts for a given state."""
#     state = request.args.get("state")
#     districts = population_data[population_data["State"] == state]["District"].unique().tolist()
#     return jsonify({"districts": districts})

# @app.route("/get_weather", methods=["POST"])
# def get_weather():
#     """Fetch current weather details for the selected district."""
#     try:
#         state = request.form["state"]
#         district = request.form["district"]

#         # Get current weather data from OpenWeatherMap API
#         response = requests.get(
#             BASE_URL,
#             params={
#                 "q": f"{district},IN",  # Assuming districts are in India
#                 "appid": API_KEY,
#                 "units": "metric",  # Temperature in Celsius
#             },
#         )
#         weather_data = response.json()

#         if response.status_code != 200:
#             return jsonify({"error": f"Error fetching weather data: {weather_data.get('message', 'Unknown error')}"})

#         # Extract current weather details
#         current_weather = {
#             "temperature": weather_data["main"]["temp"],
#             "humidity": weather_data["main"]["humidity"],
#             "rainfall": weather_data.get("rain", {}).get("1h", 0),  # Rainfall in mm in the past hour
#         }

#         return jsonify(current_weather)

#     except Exception as e:
#         return jsonify({"error": str(e)})

# @app.route("/predict", methods=["POST"])
# def predict():
#     """Handle prediction based on selected state and district."""
#     try:
#         state = request.form["state"]
#         district = request.form["district"]

#         # Get average climate data
#         district_climate = climate_data[(climate_data["State"] == state) & (climate_data["District"] == district)]
#         if district_climate.empty:
#             return jsonify({"error": "No climate data found for the selected district."})

#         avg_temperature = district_climate["AvgTemperature"].values[0]
#         avg_humidity = district_climate["AvgHumidity"].values[0]
#         avg_rainfall = district_climate["TotalRainfall"].values[0]

#         # Get population data
#         district_population = population_data[(population_data["State"] == state) & (population_data["District"] == district)]
#         if district_population.empty:
#             return jsonify({"error": "No population data found for the selected district."})

#         population = district_population["Population"].values[0]
#         population_density = district_population["PopulationDensity"].values[0]
#         urbanization_rate = district_population["UrbanizationRate"].values[0]

#         # Get healthcare data
#         district_healthcare = healthcare_data[(healthcare_data["State"] == state) & (healthcare_data["District"] == district)]
#         if district_healthcare.empty:
#             return jsonify({"error": "No healthcare data found for the selected district."})

#         hospitals = district_healthcare["Hospitals"].values[0]
#         beds = district_healthcare["Beds"].values[0]
#         doctors = district_healthcare["Doctors"].values[0]
#         nurses = district_healthcare["Nurses"].values[0]

#         # Dummy current weather (replace with real data from `/get_weather` if needed)
#         current_temperature = avg_temperature + 1
#         current_humidity = avg_humidity + 2
#         current_rainfall = avg_rainfall + 3

#         # Prepare input data
#         input_data = np.array(
#             [[
#                 (current_temperature + avg_temperature) / 2,
#                 (current_humidity + avg_humidity) / 2,
#                 (current_rainfall + avg_rainfall) / 2,
#                 population,
#                 population_density,
#                 urbanization_rate,
#                 hospitals,
#                 beds,
#                 doctors,
#                 nurses,
#             ]]
#         )
#         input_data_scaled = scaler.transform(input_data)

#         # Make prediction
#         prediction = model.predict(input_data_scaled)[0]
#         probability = model.predict_proba(input_data_scaled)[0][1]  # Probability of high outbreak

#         # Return results
#         result = {
#             "prediction": "High Outbreak Risk" if prediction == 1 else "Low Outbreak Risk",
#             "confidence": f"{probability:.2%}" if prediction == 1 else f"{1 - probability:.2%}",
#         }
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import requests

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("disease_outbreak_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load data
climate_data = pd.read_csv("climate_data_monthly.csv")
population_data = pd.read_csv("population_data.csv")
healthcare_data = pd.read_csv("healthcare_data.csv")

# OpenWeatherMap API configuration
API_KEY = "2c84a58594dea5397b1a0b140de2ebcf"  # Replace with your actual API key
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

@app.route("/")
def index():
    """Render the main page with state dropdown pre-filled."""
    states = population_data["State"].unique().tolist()
    return render_template("index.html", states=states)

@app.route("/get_districts", methods=["POST"])
def get_districts():
    """Return districts for the selected state."""
    state = request.form["state"]
    districts = population_data[population_data["State"] == state]["District"].unique().tolist()
    return jsonify({"districts": districts})

@app.route("/get_weather", methods=["POST"])
def get_weather():
    """Fetch current weather details for the selected district."""
    try:
        state = request.form["state"]
        district = request.form["district"]

        # Get current weather data from OpenWeatherMap API
        response = requests.get(
            BASE_URL,
            params={
                "q": f"{district},IN",  # Assuming districts are in India
                "appid": API_KEY,
                "units": "metric",
            },
        )
        weather_data = response.json()

        if response.status_code != 200:
            return jsonify({"error": f"Error fetching weather data: {weather_data.get('message', 'Unknown error')}"})

        # Extract current weather details
        current_weather = {
            "temperature": weather_data["main"]["temp"],
            "humidity": weather_data["main"]["humidity"],
            "rainfall": weather_data.get("rain", {}).get("1h", 0),
        }

        return jsonify(current_weather)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction logic."""
    try:
        state = request.form["state"]
        district = request.form["district"]

        # Get average climate data
        district_climate = climate_data[(climate_data["State"] == state) & (climate_data["District"] == district)]
        if district_climate.empty:
            return jsonify({"error": "No climate data found for the selected district."})

        avg_temperature = district_climate["AvgTemperature"].values[0]
        avg_humidity = district_climate["AvgHumidity"].values[0]
        avg_rainfall = district_climate["TotalRainfall"].values[0]

        # Get population data
        district_population = population_data[(population_data["State"] == state) & (population_data["District"] == district)]
        if district_population.empty:
            return jsonify({"error": "No population data found for the selected district."})

        population = district_population["Population"].values[0]
        population_density = district_population["PopulationDensity"].values[0]
        urbanization_rate = district_population["UrbanizationRate"].values[0]

        # Get healthcare data
        district_healthcare = healthcare_data[(healthcare_data["State"] == state) & (healthcare_data["District"] == district)]
        if district_healthcare.empty:
            return jsonify({"error": "No healthcare data found for the selected district."})

        hospitals = district_healthcare["Hospitals"].values[0]
        beds = district_healthcare["Beds"].values[0]
        doctors = district_healthcare["Doctors"].values[0]
        nurses = district_healthcare["Nurses"].values[0]

        # Dummy current weather (replace with real data from `/get_weather` if needed)
        current_temperature = avg_temperature + 1
        current_humidity = avg_humidity + 2
        current_rainfall = avg_rainfall + 3

        # Prepare input data
        input_data = np.array(
            [[
                (current_temperature + avg_temperature) / 2,
                (current_humidity + avg_humidity) / 2,
                (current_rainfall + avg_rainfall) / 2,
                population,
                population_density,
                urbanization_rate,
                hospitals,
                beds,
                doctors,
                nurses,
            ]]
        )
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)[0]
        probability = model.predict_proba(input_data_scaled)[0][1]

        result = {
            "prediction": "High Outbreak Risk" if prediction == 1 else "Low Outbreak Risk",
            "confidence": f"{probability:.2%}" if prediction == 1 else f"{1 - probability:.2%}",
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
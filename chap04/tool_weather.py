import requests


def get_weather(city: str) -> str:
    """
    Fetches real-time weather data for a given city using the OpenWeatherMap API.

    Args:
        city (str): The name of the city.

    Returns:
        str: Weather information in a user-friendly format or an error message.
    """
    # TODO: Replace 'your_api_key_here' with your actual OpenWeatherMap API key
    api_key = "your_api_key_here"

    # Construct the API URL with the city, API key, and set units to metric
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        # Send a GET request to the OpenWeatherMap API
        response = requests.get(url)

        # Raise an exception for HTTP error responses (e.g., 404, 500)
        response.raise_for_status()

        # Parse the JSON response into a Python dictionary
        weather_data = response.json()

        # Extract weather details from the JSON data
        weather_desc = weather_data["weather"][0][
            "description"
        ]  # Weather condition description
        temp = weather_data["main"]["temp"]  # Current temperature in °C
        humidity = weather_data["main"]["humidity"]  # Humidity percentage
        wind_speed = weather_data["wind"]["speed"]  # Wind speed in m/s

        # Format and return the weather information as a string
        return (
            f"The current weather in {city} is {weather_desc} with a temperature of {temp}°C, "
            f"humidity at {humidity}%, and wind speed of {wind_speed} m/s."
        )
    except requests.exceptions.RequestException as e:
        # In case of any HTTP request issues, return an error message
        return f"Failed to retrieve weather data: {str(e)}"

import requests


def evaluate_weather_api(query_city, expected_temperature_range):
    """
    Fetches weather data from WeatherAPI and evaluates whether the reported temperature
    falls within the expected temperature range.

    Args:
        query_city (str): The name of the city to query weather information.
        expected_temperature_range (tuple): A tuple with the expected minimum and maximum temperature (in °C).

    Returns:
        dict: A dictionary containing:
            - 'correct': Boolean indicating if the reported temperature is within the expected range.
            - 'reported_temperature': The temperature returned by the API.
            - 'expected_range': The input expected temperature range.
            - or 'error': An error message if the API response is invalid.
    """
    # Build the API URL with the provided city and API key placeholder.
    api_url = (
        f"https://api.weatherapi.com/v1/current.json?key={YOUR_API_KEY}&q={query_city}"
    )

    # Send a GET request to the WeatherAPI and parse the JSON response.
    response = requests.get(api_url).json()

    # Check if the API response contains 'current' weather data.
    if "current" in response:
        # Extract the current temperature in Celsius.
        temp = response["current"]["temp_c"]
        # Verify if the temperature is within the expected range.
        return {
            "correct": expected_temperature_range[0]
            <= temp
            <= expected_temperature_range[1],
            "reported_temperature": temp,
            "expected_range": expected_temperature_range,
        }

    # Return an error if the API response is invalid or does not contain expected data.
    return {"error": "Invalid API response"}


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
# Evaluate the weather in New York with an expected temperature range of 10°C to 25°C.
test_result = evaluate_weather_api("New York", (10, 25))
print(test_result)

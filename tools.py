from random import randint

def get_current_temperature(city: str, country: str = "United Kingdom") -> dict:
    """Gets the current temperature for a given location.

    Args:
        city: The city name
        country: The country name if known (optional - defaults to "United Kingdom")

    Returns:
        A dictionary containing the temperature in celsius, the city, and the country
    """
    return {"celsius": randint(10, 20), "city": city, "country": country}
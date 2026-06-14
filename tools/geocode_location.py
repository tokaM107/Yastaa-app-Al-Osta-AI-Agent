import requests


TOOL_SCHEMA = {
    "name": "geocode_location",
    "description": "Convert a place name in Alexandria to one best geographic coordinate pair (top-1 latitude/longitude). Always use this tool before searching for a multimodal transit route if you only have the place name.",
    "parameters": {
        "type": "object",
        "properties": {
            "place_name": {
                "type": "string",
                "description": "The name of the place (e.g., Mahatet Misr, Sidi Gaber, San Stefano)"
            },
            "user_lat": {
                "type": "number",
                "description": "Optional user latitude to bias the geocode search around the user's position"
            },
            "user_lng": {
                "type": "number",
                "description": "Optional user longitude to bias the geocode search around the user's position"
            },
            "bias": {
                "type": "boolean",
                "description": "Whether to bias the query to Alexandria before falling back to a broader search"
            }
        },
        "required": ["place_name"]
    }
}


def _geocode_request(place_name, bias=True, user_lat=None, user_lng=None):
    url = "http://localhost:8003/api/v1/geocode"
    params = {
        "address": place_name,
        "language": "en",
        "bias": bias,
    }
    if user_lat is not None:
        params["user_lat"] = user_lat
    if user_lng is not None:
        params["user_lng"] = user_lng

    response = requests.get(url, params=params, timeout=5)
    if response.status_code != 200:
        return {"error": f"Geocoding failed: {response.status_code}"}

    data = response.json()
    if not isinstance(data, dict):
        return {"error": "Unexpected geocoding response format"}

    if not data.get("success", False):
        return {"error": data.get("error", "No geocoding results found")}

    results = data.get("results", [])
    if not results:
        return {"error": "No geocoding results found"}

    top_result = results[0] if isinstance(results[0], dict) else {}
    lat = top_result.get("latitude")
    lon = top_result.get("longitude")

    if lat is None or lon is None:
        return {"error": "Top geocoding result is missing coordinates"}

    return {
        "lat": lat,
        "lon": lon,
        "formatted_address": top_result.get("formatted_address", ""),
    }


def execute_geocode(place_name, user_lat=None, user_lng=None, bias=True):
    """Call the Real Geocoding API"""
    try:
        result = _geocode_request(place_name, bias=bias, user_lat=user_lat, user_lng=user_lng)
        if "error" not in result or not bias:
            return result

        retry_place_name = f"{place_name.strip()} Corniche Alexandria" if place_name.strip() else place_name
        retry_result = _geocode_request(retry_place_name, bias=False, user_lat=user_lat, user_lng=user_lng)
        if "error" not in retry_result:
            return retry_result

        return result
    except Exception as e:
        return {"error": f"Error connecting to Geocoding API: {e}"}

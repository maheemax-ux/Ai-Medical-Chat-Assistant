

import requests
from geopy.geocoders import Nominatim

from config import OVERPASS_URL, NOMINATIM_USER_AGENT, DEFAULT_SEARCH_RADIUS_METERS


def geocode_address(address: str):
    """Returns (lat, lon) or None if not found."""
    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"[hospital_finder] geocoding failed: {e}")
    return None


def get_user_location_by_ip():
    """Get approximate user location from IP address (no browser permission needed)."""
    try:
        resp = requests.get("http://ip-api.com/json/?fields=lat,lon,city,regionName,country", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("lat") and data.get("lon"):
            city = data.get("city", "")
            region = data.get("regionName", "")
            country = data.get("country", "")
            location_name = ", ".join(p for p in [city, region, country] if p)
            return data["lat"], data["lon"], location_name
    except Exception as e:
        print(f"[hospital_finder] IP geolocation failed: {e}")
    return None


def find_nearby_hospitals(lat: float, lon: float, radius_m: int = DEFAULT_SEARCH_RADIUS_METERS):
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      relation["amenity"="hospital"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """
    try:
        resp = requests.get(
            OVERPASS_URL,
            params={"data": query.strip()},
            headers={"User-Agent": "MedicalAssistantApp/1.0"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[hospital_finder] Overpass query failed: {e}")
        return []

    hospitals = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "Unnamed Hospital")

        if el["type"] == "node":
            h_lat, h_lon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center", {})
            h_lat, h_lon = center.get("lat"), center.get("lon")

        if h_lat is None or h_lon is None:
            continue

        address_parts = [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:city", ""),
        ]
        address = ", ".join(p for p in address_parts if p) or "Address not available"

        hospitals.append({
            "name": name,
            "lat": h_lat,
            "lon": h_lon,
            "address": address,
            "phone": tags.get("phone", tags.get("contact:phone", "N/A")),
        })

    return hospitals

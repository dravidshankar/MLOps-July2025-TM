import requests

ride = {
    "trip_distance": 4.6,
    "trip_duration": 900
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print("Prediction response:", response.json())

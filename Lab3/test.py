import requests

ride = {
    "trip_distance": 3.2,
    "trip_duration": 720
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print("Prediction response:", response.json())

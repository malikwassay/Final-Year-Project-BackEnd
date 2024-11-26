import requests

url = "http://127.0.0.1:5000/predict_both"
payload = {"city": "iquitos"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    print("API Response:", response.json())
else:
    print("Error:", response.status_code, response.text)

import requests
import json

BASE_URL = "http://localhost:5000"

def test_predict():
    print("Testing /predict...")
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200

def test_model_info():
    print("\nTesting /model/info...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200

def test_stats_json():
    print("\nTesting /stats?format=json...")
    payload = {"data": [10, 20, 30, 40, 100]}
    response = requests.post(f"{BASE_URL}/stats?format=json", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200

def test_stats_image():
    print("\nTesting /stats?plot_type=box...")
    payload = {"data": [10, 20, 30, 40, 100]}
    response = requests.post(f"{BASE_URL}/stats?plot_type=box", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        with open("tests/test_boxplot.png", "wb") as f:
            f.write(response.content)
        print("Image saved to tests/test_boxplot.png")
    assert response.status_code == 200

if __name__ == "__main__":
    # Note: API must be running for these tests to work
    try:
        test_predict()
        test_model_info()
        test_stats_json()
        test_stats_image()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")

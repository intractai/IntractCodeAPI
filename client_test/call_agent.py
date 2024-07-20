import argparse
import pathlib
import requests

USER_NAME = 'erfan_miahi'
PASS = 'temp'

def login(port: int = 8000):
    # data = {'username': USER_NAME, 'email': 'mhi.erfan1@gmail.com', 'password': PASS}
    # response = requests.post(f'http://localhost:{port}/register', json=data)
    data = {'username': USER_NAME, 'password': PASS}
    form_data = {
        'username': USER_NAME,
        'password': PASS
    }

    # Headers to be sent in the POST request
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    response = requests.post(f'http://localhost:{port}/token', data=form_data, headers=headers)
    if response.status_code == 200:
        print('Request successful!')
        # Accessing the response content
        print('Response content:', response.json())
        return response.json()['access_token']
    else:
        print('Request failed with status code:', response.status_code)
        print('Response content:', response.text)
        return None


def send_request(input_text, proceeding_text, port=8000):
    token = login(port)
    if not token:
        print("Failed to retrieve token. Exiting.")
        return
    
    # Headers to be sent in the POST request
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    print('Headers: ', headers)

    # Ensure the endpoint and method are correct (change PUT to POST if necessary)
    response = requests.post(f"http://localhost:{port}/generate", json={
        "file_path": str(pathlib.Path(__file__).parent.absolute()),  # This file's path
        "prior_context": input_text,
        "proceeding_context": proceeding_text,
        "max_decode_length": 128,
    }, timeout=180, headers=headers)

    # Check for successful request
    if response.status_code == 200:
        try:
            # Check if the response is JSON
            if 'application/json' in response.headers.get('Content-Type', ''):
                output_data = response.json()
                if "error" in output_data:
                    print(f"Error: {output_data['error']}")
                    return
                output_text = output_data["generated_text"]
                score = output_data["score"]

                print("Input text: " + input_text)
                print(f"Generated text ({score:.3f}):")
                print(output_text)
            else:
                print("Response is not JSON:")
                print(response.text)
        except ValueError as e:
            print('Error decoding JSON:', e)
            print('Response content:', response.text)
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(f"Response content: {response.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send a request to the API.')
    parser.add_argument('input_text', type=str,
        help='The input text to send to the API.')
    parser.add_argument('--proceeding_text', type=str, default=None,
        help='The proceeding text for FIM')
    parser.add_argument('--port', type=int, default=8000,
        help='Port for the API.')
    args = parser.parse_args()

    send_request(args.input_text, args.proceeding_text, args.port)

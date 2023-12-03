import argparse
import requests


def send_request(input_text, port=8000):
    response = requests.put(f"http://localhost:{port}/generate", json={"input_text": input_text})
    output_data = response.json()
    output_text = output_data["generated_text"]
    score = output_data["score"]

    print("Input text: " + input_text)
    print(f"Generated text ({score:.3f}):")
    print(output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send a request to the API.')
    parser.add_argument('input_text', type=str, help='The input text to send to the API.')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API.')
    args = parser.parse_args()
    send_request(args.input_text, args.port)

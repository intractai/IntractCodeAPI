import argparse
import pathlib
import requests


def send_request(input_text, proceeding_text, port=8000):
    response = requests.put(f"http://localhost:{port}/generate", json={
        "file_path": str(pathlib.Path(__file__).parent.absolute()), # This file's path
        "prior_context": input_text,
        "proceeding_context": proceeding_text,
        "max_decode_length": 128,
    }, timeout=180)
    output_data = response.json()
    if "error" in output_data:
        print(f"Error: {output_data['error']}")
        return
    output_text = output_data["generated_text"]
    score = output_data["score"]

    print("Input text: " + input_text)
    print(f"Generated text ({score:.3f}):")
    print(output_text)


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

import requests
import argparse

def send_request(input_text):
    response = requests.put("http://localhost:8000/generate", json={"input_text": input_text})
    output_data=response.json()
    print("Input text: " + input_text)
    print("Generated text:")
    print(str(output_data["generated_text"].strip()))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send a request to the API.')
    parser.add_argument('input_text', type=str, help='The input text to send to the API.')
    args = parser.parse_args()
    send_request(args.input_text)
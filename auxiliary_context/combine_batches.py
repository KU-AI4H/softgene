from argparse import ArgumentParser
import json

argparser = ArgumentParser()
argparser.add_argument('files', nargs='+')

DATA_PATH = Path("data")
OUTPUT_PATH = DATA_PATH / "responses.json"

def combine_batches(paths: list, output_path: str):
    parsed_responses = {}
    for path in paths:
        with open(path, "r", encoding='utf-8') as f:
            responses = json.load(f)
            for response in responses:
                parsed_responses[response['custom_id'].replace("GO_", "GO:")] = response['response']['body']['output'][0]['content'][0]['text']
    with open(output_path, "w") as f:
        json.dump(parsed_responses, f, indent=2)

def main():
    combine_batches(argparser.parse_args().files, OUTPUT_PATH)

if __name__ == "__main__":
    main()
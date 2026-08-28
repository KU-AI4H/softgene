from argparse import ArgumentParser
from openai import OpenAI

argparser = ArgumentParser()
argparser.add_argument('batch_id')

def check_batch_status(batch_id: str):
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    print(batch.status)

if __name__ == "__main__":
    args = argparser.parse_args()
    batch_id = args.batch_id
    check_batch_status(batch_id)
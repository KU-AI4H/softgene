from openai import OpenAI
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument("batch_id")

def debug_batch(batch_id: str):
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    print("Status:", batch.status)
    print("Errors:", batch.errors)
    print("Error file:", batch.error_file_id)

    if batch.error_file_id:
        content = client.files.content(batch.error_file_id)
        print("\n--- Error file contents ---")
        print(content.decode("utf-8"))

debug_batch(argparser.parse_args().batch_id)
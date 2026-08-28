from openai import OpenAI
from argparse import ArgumentParser
import json

argparser = ArgumentParser()
argparser.add_argument("batch_id", help="ID of the completed batch to download")
args = argparser.parse_args()

client = OpenAI()

def download_batch(batch_id: str):
    # Retrieve batch info
    batch = client.batches.retrieve(batch_id)
    print("Batch status:", batch.status)

    if batch.status != "completed":
        print("Batch is not yet completed.")
        return

    all_outputs = []

    # Attempt 1: check if batch has output_file_id
    output_file_id = getattr(batch, "output_file_id", None)
    if output_file_id:
        print(f"Downloading output file {output_file_id} ...")
        content = client.files.content(output_file_id)
        print(content.text[:20])
        for line in content.text.splitlines():
            item = json.loads(line)
            all_outputs.append({
                "custom_id": item.get("custom_id"),
                "response": item.get("response")
            })
    else:
        # Attempt 2: retrieve results via list_results (SDK method may differ)
        print("No output_file_id found; fetching results directly...")
        results = client.batches.results.list(batch_id)
        for r in results.data:
            custom_id = r.get("custom_id")
            response = r.get("response", {})
            output_text = response.get("output_text") or response.get("body", {}).get("output_text")
            all_outputs.append({"custom_id": custom_id, "output": output_text})

    # Save to JSON file
    output_path = f"{batch_id}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {len(all_outputs)} results to '{output_path}'")

if __name__ == "__main__":
    download_batch(args.batch_id)

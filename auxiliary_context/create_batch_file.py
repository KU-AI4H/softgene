import json
from pathlib import Path
from math import ceil

INPUT_GENE_SET_FILE = r"data\msigdb\gene_sets_merged.json"
BATCH_OUTPUT_DIR = Path(r"data\gpt_summaries\msigdb")

MAX_RECORDS_PER_BATCH = 4000

SYSTEM_PROMPT = """You are a computational biologist writing Gene Ontology–style functional summaries.
Use established biological terminology and avoid speculation.
Prefer concise, high-level functional descriptions over gene-specific details.
Do not list individual genes unless functionally necessary."""

USER_PROMPT = """Summarize the shared biological function of the following genes in 2–3 concise sentences.
Focus on biological processes, molecular functions, or cellular components that are well-established in the literature.
Use precise biological terminology suitable for Gene Ontology annotation.

Gene symbols:
{gene_symbols}"""

def form_prompt(gene_symbols: list[str]) -> str:
    return USER_PROMPT.format(
        gene_symbols=",".join(gene_symbols[:25])
    )

def sanitize_custom_id(identifier: str) -> str:
    return identifier.replace(":", "_")

def create_batched_files(
    gene_set_file: str | Path,
    output_dir: str | Path,
    max_records_per_batch: int = MAX_RECORDS_PER_BATCH,
):
    gene_set_file = Path(gene_set_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(gene_set_file, "r") as f:
        gene_sets = json.load(f)

    identifiers = list(gene_sets.keys())
    total = len(identifiers)
    num_batches = ceil(total / max_records_per_batch)

    print(f"Loaded {total} gene sets from '{gene_set_file}'.")
    print(f"Writing {num_batches} batch files ({max_records_per_batch} records each).")

    for batch_idx in range(num_batches):
        start = batch_idx * max_records_per_batch
        end = min(start + max_records_per_batch, total)

        batch_path = output_dir / f"request_batch_{batch_idx:03d}.jsonl"
        count = 0

        with open(batch_path, "w") as out:
            for identifier in identifiers[start:end]:
                gene_set = gene_sets[identifier]

                request = {
                    "custom_id": sanitize_custom_id(identifier),
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": "gpt-4.1-nano",
                        "input": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": form_prompt(gene_set["genes"])}
                        ],
                        "max_output_tokens": 100
                    }
                }

                out.write(json.dumps(request) + "\n")
                count += 1

        print(f"Wrote {count} requests → '{batch_path}'")

    print("🚀 All batch files ready for submission.")

if __name__ == "__main__":
    create_batched_files(INPUT_GENE_SET_FILE, BATCH_OUTPUT_DIR)

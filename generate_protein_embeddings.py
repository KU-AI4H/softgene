from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from Bio import SeqIO
from pathlib import Path

DATA_PATH = Path("data")
EMBEDDINGS_PATH = DATA_PATH / "embeddings"
FASTA_PATH = DATA_PATH / "Homo_sapiens.GRCh38.pep.all_clean.fa"
OUTPUT_PATH = EMBEDDINGS_PATH / "protein_embeddings.pt"
PROTEIN_ENCODER_MODEL_NAME = "facebook/esm2_t33_150M_UR50D"
BATCH_SIZE = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_protein_encoder(model_name: str) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer

def load_protein_sequences(path: str | Path):
    records = list(SeqIO.parse(FASTA_PATH, "fasta"))
    print(f"Found {len(records)} protein sequences.")
    return records

def embed_protein_sequences(
    encoder, tokenizer, sequences, batch_size: int = 16
) -> dict:
    embeddings = {}

    for i in tqdm(range(0, len(sequences), BATCH_SIZE)):
        batch = sequences[i : i + BATCH_SIZE]

        # Prepare input texts
        batch_seqs = [str(rec.seq) for rec in batch]
        batch_ids = [rec.id for rec in batch]

        # Tokenize
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=1024
        ).to(device)

        # Run model
        with torch.no_grad():
            outputs = encoder(**inputs)

        # Get embeddings — use mean over residues (excluding padding)
        token_embs = outputs.last_hidden_state
        attention_mask = inputs.attention_mask.unsqueeze(-1)
        masked = token_embs * attention_mask
        summed = masked.sum(dim=1)
        lengths = attention_mask.sum(dim=1)
        mean_embs = summed / lengths

        # Save to dict (move to CPU to save GPU memory)
        for pid, emb in zip(batch_ids, mean_embs):
            embeddings[pid] = emb.cpu()

        del inputs, outputs, token_embs, masked, summed, mean_embs, attention_mask, lengths

    return embeddings

def generate_protein_embeddings(
    fasta_path: str | Path,
    encoder_model_name: str,
    output_path: str | Path,
    batch_size: float = 16
) -> dict:
    encoder, tokenizer = load_protein_encoder(encoder_model_name)
    sequences = load_protein_sequences(fasta_path)
    embeddings = embed_protein_sequences(encoder, tokenizer, sequences, batch_size)
    return embeddings

def main():
    embeddings = generate_protein_embeddings(
        FASTA_PATH, PROTEIN_ENCODER_MODEL_NAME, OUTPUT_PATH, BATCH_SIZE
    )
    torch.save(embeddings, OUTPUT_PATH)

if __name__ == "__main__":
    main()
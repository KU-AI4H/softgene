import torch
from pathlib import Path
import json

DATA_FOLDER_PATH = Path("data")

EMBEDDINGS_PATH = DATA_FOLDER_PATH / "embeddings"
MAPPINGS_PATH = DATA_FOLDER_PATH / "mappings"

PROTEIN_EMBEDDINGS_PATH = EMBEDDINGS_PATH / "protein_embeddings.pt"
GENE_TO_PROTEINS_PATH = DATA_FOLDER_PATH / "Homo_sapiens.GRCh38.gene_symbol_to_protein_ID.json"
GENE_SETS_PATH = DATA_FOLDER_PATH / "go" / "go_gene_sets_all.json" # Change to "msigdb" / "msigdb_gene+sets_all.json" to train/evaluate on MSigDB

EMBEDDINGS_OUTPUT_PATH = "embeddings"
MAPPINGS_OUTPUT_PATH = "mappings"

EMBEDDINGS_OUTPUT_PATH.mkdir(parents=True,exist_ok=True)
MAPPINGS_OUTPUT_PATH.mkdir(parents=True,exist_ok=True)

STACKED_PROTEIN_EMBEDDINGS_OUTPUT_PATH = EMBEDDINGS_OUTPUT_PATH / "gene_embedding_stacked.pt"
PROTEIN_INDICES_OUTPUT_PATH = MAPPINGS_OUTPUT_PATH / "protein_gene_index.pt"
GENE_INDICES_FLAT_PATH = MAPPINGS_OUTPUT_PATH / "gene_indices_flat.pt"
SET_INDICES_FLAT_PATH = MAPPINGS_OUTPUT_PATH / "set_indices_flat.pt"

PROTEIN_ID_MAPPING_PATH = MAPPINGS_OUTPUT_PATH / "protein_to_idx.json"
GENE_ID_MAPPING_PATH = MAPPINGS_OUTPUT_PATH / "gene_to_idx.json"
SET_ID_MAPPING_PATH = MAPPINGS_OUTPUT_PATH / "set_to_idx.json"

def generate_vectorized_ready_embeddings():
    print("Loading protein embeddings...")
    protein_embeddings = torch.load(PROTEIN_EMBEDDINGS_PATH)
    print(f"Loaded {len(protein_embeddings)} protein embeddings.")

    with open (GENE_TO_PROTEINS_PATH, "r") as genes_file:
        gene_to_proteins = json.load(genes_file)

    with open(GENE_SETS_PATH, "r") as gene_set_file:
        gene_set_mappings = json.load(gene_set_file)

    set_to_genes = {symbol: set['genes'] for symbol, set in gene_set_mappings.items()}

    print("Building mappings...")
    all_proteins = list(protein_embeddings.keys())
    all_genes = list(gene_to_proteins.keys())
    all_sets = list(set_to_genes.keys())

    # protein_to_idx = {pid: i for i, pid in enumerate(all_proteins)}
    gene_to_idx = {gid: i for i, gid in enumerate(all_genes)}
    set_to_idx = {sid: i for i, sid in enumerate(all_sets)}

    PROTEIN_ID_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(PROTEIN_ID_MAPPING_PATH, "w") as f:
        json.dump(protein_to_idx, f)
    
    with open(GENE_ID_MAPPING_PATH, "w") as f:
        json.dump(gene_to_idx, f)

    with open(SET_ID_MAPPING_PATH, "w") as f:
        json.dump(set_to_idx, f)
    print(f"Mapping '.json' files saved to ./mappings/")

    print(f"Stacking protein tensors...")
    protein_to_idx = {}
    stacked_protein_tensors = []
    protein_gene_indices_list = []
    current_idx = 0

    for gene_symbol, protein_ids in gene_to_proteins.items():
        gene_idx = gene_to_idx[gene_symbol]
        for pid in protein_ids:
            if pid in protein_embeddings:
                protein_to_idx[pid] = current_idx  # index matches stacking order
                stacked_protein_tensors.append(protein_embeddings[pid])
                protein_gene_indices_list.append(gene_idx)
                current_idx += 1

    with open(PROTEIN_ID_MAPPING_PATH, "w") as f:
        json.dump(protein_to_idx, f)
    exit()

    all_protein_embeddings = torch.stack(stacked_protein_tensors)
    protein_gene_index = torch.tensor(protein_gene_indices_list, dtype=torch.long)

    torch.save(all_protein_embeddings, STACKED_PROTEIN_EMBEDDINGS_OUTPUT_PATH)
    torch.save(protein_gene_index, PROTEIN_INDICES_OUTPUT_PATH)
    print("Stacked tensors and indices saved to ./mappings/")

    print("Getting gene indices...")

    gene_indices_flat_list = []
    set_indices_flat_list = []

    num_skipped = 0

    for set_symbol, gene_symbols in set_to_genes.items():
        set_idx = set_to_idx[set_symbol]
        for gene_symbol in gene_symbols:
            if gene_symbol in gene_to_idx:
                gene_idx = gene_to_idx[gene_symbol]

                gene_indices_flat_list.append(gene_idx)
                set_indices_flat_list.append(set_idx)
            else:
                print(f"Skipping {gene_symbol} in gene set {set_symbol}")
                num_skipped += 1

    print(f"Total genes skipped: {num_skipped / len(set_to_genes)}")

    gene_indices_flat = torch.tensor(gene_indices_flat_list, dtype=torch.long)
    set_indices_flat = torch.tensor(set_indices_flat_list, dtype=torch.long)

    torch.save(gene_indices_flat, GENE_INDICES_FLAT_PATH)
    torch.save(set_indices_flat, SET_INDICES_FLAT_PATH)
    print(f"Gene indices saved to {GENE_INDICES_FLAT_PATH}.parent.")

if __name__ == "__main__":
    generate_vectorized_ready_embeddings()
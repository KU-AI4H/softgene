from Bio import SeqIO
from pathlib import Path
import json
import re

FASTA_PATH = Path("Homo_sapiens.GRCh38.pep.all_clean.fa")
OUTPUT_PATH = Path("data") / "Homo_sapiens.GRCh38.gene_symbol_to_protein_ID.json"

gene_to_proteins = {}

for record in SeqIO.parse(FASTA_PATH, "fasta"):
    # Header format: ENSP00000363092.5 pep ... gene_symbol:PRKG1 ...
    match = re.search(r'gene_symbol:(\S+)', record.description)
    if match:
        gene_symbol = match.group(1)
        protein_id = record.id
        gene_to_proteins.setdefault(gene_symbol, []).append(protein_id)

with open(OUTPUT_PATH, "w") as f:
    json.dump(gene_to_proteins, f, indent=2)

print(f"Done. {len(gene_to_proteins)} genes written.")
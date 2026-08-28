# SoftGene

**SoftGene** is a framework for LLM-based gene set annotation that grounds language model generation in protein sequence structure. A hierarchical attention encoder aggregates ESM-2 protein embeddings at the isoform, gene, and gene set levels, and projects the resulting representation into soft prompt tokens that condition a frozen local LLM alongside LLM-generated auxiliary context.

> Paper: *SoftGene: Protein Language Model-Enhanced Soft Prompting for Interpretable Gene Set Annotation* (EMNLP 2026 Main Conference)

## Installation

```bash
git clone ...
cd softgene
pip install -r requirements.txt
```

## Quickstart

### Step 1 — Generate Protein Embeddings

First, download the [Ensembl human proteome FASTA](https://uswest.ensembl.org/info/data/ftp/index.html
). Then, use [this script](https://github.com/snap-stanford/SATURN/blob/main/protein_embeddings/clean_fasta.py) from the SATURN project to clean the file and save it as:


```
Homo_sapiens.GRCh38.pep.all_clean.fa
```

Then run:

```bash
python generate_protein_embeddings.py
```

This produces `data/embeddings/protein_embeddings.pt` — a dictionary mapping Ensembl protein IDs to ESM-2 mean-pooled embeddings. By default this uses `facebook/esm2_t33_150M_UR50D` but can be configured by changing the `PROTEIN_ENCODER_MODEL_NAME` to a different HuggingFace model checkpoint. This step is slow but only needs to be run once; embeddings can be cached and reused across experiments.

Finally, run:

```bash
python build_gene_protein_map.py
```

This produces `Homo_sapiens.GRCh38.gene_symbol_to_protein_ID.json`, a mapping from each gene symbol to all corresponding splice isoforms.

### Step 2 — Build Mappings

```bash
python build_stacked_embeddings.py
```

This reads the protein embeddings, the gene symbol → protein ID mapping, and the gene set JSON, and produces the following files:

```
data/
├── embeddings/
│   └── gene_embedding_stacked.pt       # Stacked protein embedding tensor
└── mappings/
    ├── protein_gene_index.pt           # Maps each protein to its gene index
    ├── gene_indices_flat.pt            # Maps genes to gene sets (flat)
    ├── set_indices_flat.pt             # Parallel set index tensor
    ├── protein_to_idx.json
    ├── gene_to_idx.json
    └── set_to_idx.json
```

### Step 3 — Generate Auxiliary Context

Generate LLM context summaries for each gene set using a commercial LLM. This produces `data/responses.json`, a dictionary mapping gene set IDs to short functional summaries. See the `auxiliary_context` directory for more instructions.

The prompt used for context generation is shown in the paper (Figure 3). We used GPT-4.1 Nano in our experiments, but any instruction-following LLM can be substituted.

### Step 4 — Train

```bash
python train.py \
    --seed 42 \
    --model-id unsloth/Meta-Llama-3.1-8B \
    --dataset-path data/go/go_gene_sets_all.json \
    --gpt-responses-path data/responses.json \
    --data-path data/ \
    --soft-prompt-length 10 \
    --num-epochs 1 \
    --batch-size 1 \
    --output-path outputs/
```

Checkpoints are saved to `outputs/seed_<seed>/best_checkpoint.pt` after each epoch.

**Multi-GPU training** is supported via Accelerate:

```bash
accelerate launch train.py \
    --seed 42 \
    --model-id unsloth/Meta-Llama-3.1-8B \
    --dataset-path data/go/go_gene_sets_all.json \
    --gpt-responses-path data/responses.json \
    --data-path data/
```

### Step 5 — Evaluate

```bash
python eval_model.py \
    --seed 42 \
    --model-path outputs/seed_42 \
    --model-id unsloth/Meta-Llama-3.1-8B \
    --data-path data/ \
    --gene-sets data/go/go_gene_sets_all.json \
    --llm-context data/responses.json \
    --soft-prompt-length 10 \
    --batch-size 4 \
    --output-path outputs/seed_42/
```

Results are saved to `outputs/seed_42/results.json` and printed to stdout.

The test split is reconstructed deterministically from the seed, matching the split used during training. Make sure `--seed` and `--soft-prompt-length` match your training run.

## Key Arguments

### Training (`train.py`)

| Argument | Default | Description |
|---|---|---|
| `--model-id` | `unsloth/Meta-Llama-3.1-8B` | HuggingFace LLM backbone |
| `--soft-prompt-length` | `10` | Number of soft prompt tokens |
| `--num-epochs` | `10` | Training epochs |
| `--batch-size` | `1` | Batch size per GPU |
| `--seed` | *(required)* | Random seed |
| `--omit-gpt-context` | `False` | Ablation: remove auxiliary context |
| `--randomize-embeddings` | `False` | Ablation: replace ESM-2 embeddings with random noise |

### Evaluation (`eval_model.py`)

| Argument | Default | Description |
|---|---|---|
| `--model-path` | *(required)* | Path to checkpoint directory (`outputs/seed_<seed>/`) |
| `--model-id` | *(required)* | Must match the model used during training |
| `--seed` | *(required)* | Must match the training seed to reproduce the test split |
| `--soft-prompt-length` | `10` | Must match the value used during training |
| `--batch-size` | `4` | Inference batch size |
| `--omit-gpt-context` | `False` | Ablation: remove auxiliary context |
| `--randomize-embeddings` | `False` | Ablation: replace ESM-2 embeddings with random noise |

## Supported LLM Backbones

SoftGene has been evaluated with the following backbones:

| Model | HuggingFace ID |
|---|---|
| Llama 3.1 8B | `unsloth/Meta-Llama-3.1-8B` |
| Llama 3.2 3B | `unsloth/Llama-3.2-3B` |
| Llama 2 13B | `unsloth/llama-2-13b` |
| Mistral NeMo 8B | `nvidia/Mistral-NeMo-Minitron-8B-Base` |
| Qwen3 8B | `unsloth/Qwen3-8B` |

## Reproducing Paper Results

To reproduce the results reported in the paper, run training and evaluation across three seeds and average:

```bash
for SEED in 42 24 12; do
    python train.py --seed $SEED \
        --model-id unsloth/Meta-Llama-3.1-8B \
        --dataset-path data/go/go_gene_sets_all.json \
        --gpt-responses-path data/responses.json \
        --data-path data/ \
        --num-epochs 1 \
        --output-path outputs/

    python eval_model.py --seed $SEED \
        --model-path outputs/seed_$SEED \
        --model-id unsloth/Meta-Llama-3.1-8B \
        --data-path data/ \
        --gene-sets data/go/go_gene_sets_all.json \
        --llm-context data/responses.json \
        --output-path outputs/eval/seed_$SEED/
done
```

## License

Apache-2.0

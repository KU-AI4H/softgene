import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch_scatter import scatter_sum, scatter_softmax
import random
import os
import numpy as np

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"All seeds set to {seed}.")

class AttentionPoolingGSE(nn.Module):
    def __init__(self, protein_emb_dim, protein_attention_hidden_dim=512, gene_attention_hidden_dim=512):
        super().__init__()
        self.protein_attention_mlp = nn.Sequential(
            nn.Linear(protein_emb_dim, protein_attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(protein_attention_hidden_dim, 1)
        )
        self.gene_attention_mlp = nn.Sequential(
            nn.Linear(protein_emb_dim, gene_attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(gene_attention_hidden_dim, 1)
        )

    def forward(self, protein_embeddings, protein_gene_index, gene_indices_flat, set_indices_flat, num_genes, num_sets):
        attn_scores_p = self.protein_attention_mlp(protein_embeddings)
        weights_p = scatter_softmax(attn_scores_p, protein_gene_index, dim=0)
        gene_embeddings = scatter_sum(weights_p * protein_embeddings, protein_gene_index, dim=0, dim_size=num_genes)

        flat_gene_embs = gene_embeddings.index_select(0, gene_indices_flat.squeeze(-1))
        attn_scores_g = self.gene_attention_mlp(flat_gene_embs)
        weights_g = scatter_softmax(attn_scores_g, set_indices_flat, dim=0)
        set_embeddings = scatter_sum(weights_g * flat_gene_embs, set_indices_flat, dim=0, dim_size=num_sets)

        return set_embeddings


class EndToEndSoftPromptLLM(nn.Module):
    def __init__(self, llm_name, soft_prompt_len, protein_emb_dim, freeze_llm=True):
        super().__init__()
        self.encoder = AttentionPoolingGSE(protein_emb_dim=protein_emb_dim)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.embedding_dim = self.llm.model.embed_tokens.embedding_dim
        
        self.projector = nn.Sequential(
            nn.Linear(protein_emb_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, soft_prompt_len * self.embedding_dim),
            nn.LayerNorm(soft_prompt_len * self.embedding_dim),
            nn.Dropout(0.1)
        )
        self.soft_prompt_len = soft_prompt_len
        if freeze_llm:
            self.llm.requires_grad_(False)

    def forward(self, protein_data, set_batch_indices, input_ids, attention_mask, labels):
        p_emb, p_idx, g_flat, s_flat, n_genes, n_sets = protein_data
        device = p_emb.device
        
        B = input_ids.size(0)

        local_indices = torch.as_tensor(set_batch_indices, device=device)

        batch_gene_mask = torch.isin(s_flat, local_indices)
        unique_genes, g_flat_batch = torch.unique(g_flat[batch_gene_mask], return_inverse=True)
        unique_sets, s_flat_batch = torch.unique(s_flat[batch_gene_mask], return_inverse=True)

        batch_protein_mask = torch.isin(p_idx, unique_genes)
        p_emb_batch = p_emb[batch_protein_mask]
        p_idx_batch = torch.searchsorted(unique_genes, p_idx[batch_protein_mask])

        batch_set_embeddings = self.encoder(
            p_emb_batch, p_idx_batch, g_flat_batch, s_flat_batch,
            unique_genes.size(0), local_indices.size(0)
        )

        soft_prompts = self.projector(batch_set_embeddings).view(B, self.soft_prompt_len, self.embedding_dim)

        text_embeds = self.llm.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([soft_prompts, text_embeds], dim=1)

        soft_attn = torch.ones((B, self.soft_prompt_len), device=device)
        full_mask = torch.cat([soft_attn, attention_mask], dim=1)
        
        soft_labels = torch.full((B, self.soft_prompt_len), -100, device=device)
        full_labels = torch.cat([soft_labels, labels], dim=1)

        return self.llm(
            inputs_embeds=inputs_embeds, 
            attention_mask=full_mask, 
            labels=full_labels, 
            use_cache=False
        ).loss

class GeneSetDataset(Dataset):
    def __init__(self, set_indices, input_texts, target_texts, tokenizer):
        self.set_indices = set_indices
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer

    def __len__(self): return len(self.set_indices)

    def __getitem__(self, index):
        input_ids = self.tokenizer(self.input_texts[index], add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(self.target_texts[index], add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        full_ids = torch.tensor(input_ids + target_ids, dtype=torch.long)
        labels = torch.tensor([-100]*len(input_ids) + target_ids, dtype=torch.long)
        return {"set_index": self.set_indices[index], "input_ids": full_ids, "labels": labels}

def collate_fn(batch):
    set_indices = torch.tensor([item["set_index"] for item in batch], dtype=torch.long)
    input_ids = nn.utils.rnn.pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = nn.utils.rnn.pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {"set_indices": set_indices, "input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

@torch.no_grad()
def evaluate(model, dataloader, protein_data):
    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        loss = model(protein_data, batch["set_indices"], batch["input_ids"], batch["attention_mask"], batch["labels"])
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)

def save_checkpoint(model, optimizer, epoch, val_loss, seed, save_dir):
    save_dir.mkdir(exist_ok=True, parents=True)
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "seed": seed
    }, save_dir / "best_checkpoint.pt")

def freeze_module(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_module(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def train_soft_prompt_model(seed, output_path, model_id, batch_size, num_epochs, soft_prompt_length, omit_gpt_context, llm_context, gene_sets_path, emb_path, map_path, skip_pretraining=False, freeze_encoder_initially=False, randomize_embeddings=False):
    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Load mappings and protein data
    protein_embeddings = torch.load(emb_path / "gene_embedding_stacked.pt")
    target_embeddings_T = torch.load(emb_path / "target_annotation_embeddings.pt")
    protein_gene_index = torch.load(map_path / "protein_gene_index.pt").view(-1).long()
    gene_indices_flat = torch.load(map_path / "gene_indices_flat.pt").view(-1).long()
    set_indices_flat = torch.load(map_path / "set_indices_flat.pt").view(-1).long()
    with open(map_path / "gene_to_idx.json", 'r') as f: num_genes = len(json.load(f))
    with open(map_path / "set_to_idx.json", 'r') as f: set_to_idx = json.load(f)
    
    if randomize_embeddings:
        protein_embeddings = torch.randn_like(protein_embeddings).to(accelerator.device)

    protein_data = (protein_embeddings, protein_gene_index, gene_indices_flat, set_indices_flat, num_genes, len(set_to_idx))

    # Prepare dataset
    with open(gene_sets_path, "r") as f: gene_sets = json.load(f)
    with open(llm_context, "r") as f: gpt_res = json.load(f)
    
    set_ids = list(gene_sets.keys())
    prompts, targets = [], []
    for sid in set_ids:
        genes = ', '.join(gene_sets[sid]['genes'][:25])
        ctx = gpt_res[sid] if not omit_gpt_context else ""
        prompts.append(f"Input: Genes: {genes}. {ctx}\nTask: Predict GO term.\nOutput:")
        targets.append(gene_sets[sid]['description'])

    indices = list(range(len(set_ids)))
    tr_idx, te_idx = train_test_split(indices, test_size=0.3, random_state=seed)

    train_loader = DataLoader(GeneSetDataset([indices[i] for i in tr_idx], [prompts[i] for i in tr_idx], [targets[i] for i in tr_idx], tokenizer), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(GeneSetDataset([indices[i] for i in te_idx], [prompts[i] for i in te_idx], [targets[i] for i in te_idx], tokenizer), batch_size=batch_size, collate_fn=collate_fn)

    model = EndToEndSoftPromptLLM(model_id, soft_prompt_length, protein_embeddings.shape[-1])
    model.to(accelerator.device)

    model.llm.gradient_checkpointing_enable()

    # Only optimize encoder and projector
    trainable_params = list(model.encoder.parameters()) + list(model.projector.parameters())
    optimizer = optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "name": "encoder",
            },
            {
                "params": model.projector.parameters(),
                "lr": 5e-4,
                "weight_decay": 1e-2,
                "name": "projector",
            }
        ]
    )
    
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    best_val_loss = float("inf")
    save_dir = Path(output_path) / f"seed_{seed}"

    for epoch in range(num_epochs):
        steps = 0
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(progress):
            with accelerator.accumulate(model):
                loss = model(
                    protein_data,
                    batch["set_indices"],
                    batch["input_ids"],
                    batch["attention_mask"], 
                    batch["labels"]
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients: accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                progress.set_postfix({"loss": f"{total_loss / (i+1):.4f}"})
            steps += 1
        save_checkpoint(accelerator.unwrap_model(model), optimizer, epoch+1, total_loss, seed, save_dir)
        print("Checkpoint Saved.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-s', '--soft-prompt-length', type=int, default=10)
    parser.add_argument('-o', '--output-path', default="./outputs/results/")
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-m', '--model-id', default="unsloth/Meta-Llama-3.1-8B")
    parser.add_argument('-e', '--seed', type=int, required=True)
    parser.add_argument('-n', '--num-epochs', type=int, default=10)
    parser.add_argument('-c', '--omit-gpt-context', action='store_true')
    parser.add_argument('-k', '--skip-pretraining', action='store_true')
    parser.add_argument('-f', '--freeze-encoder-initially', action='store_true', default=False)
    parser.add_argument('-d', '--dataset-path')
    parser.add_argument('-r', '--gpt-responses-path')
    parser.add_argument('-a', '--data-path', default='./data')
    parser.add_argument('-z', '--randomize-embeddings', action='store_true', default=False)
    args = parser.parse_args()

    set_all_seeds(args.seed)

    print("------PARAMETERS------")
    print(f"Soft Prompt Length: {args.soft_prompt_length}")
    print(f"Output Path: {args.output_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Model ID: {args.model_id}")
    print(f"Seed: {args.seed}")
    print(f"# of Epochs: {args.num_epochs}")
    print(f"Omit GPT Context: {args.omit_gpt_context}")
    print(f"Skip Pretraining: {args.skip_pretraining}")
    print(f"Dataset Path: {args.dataset_path}")
    print(f"LLM Responses Path: {args.gpt_responses_path}")
    print(f"Embeddings randomized: {args.randomize_embeddings}")

    DATA_PATH = Path(args.data_path)
    train_soft_prompt_model(
        seed=args.seed,
        output_path=args.output_path,
        model_id=args.model_id,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        soft_prompt_length=args.soft_prompt_length,
        omit_gpt_context=args.omit_gpt_context,
        llm_context=args.gpt_responses_path or DATA_PATH / "responses.json",
        gene_sets_path=args.dataset_path or DATA_PATH / "go_gene_sets_all.json",
        emb_path=DATA_PATH / "embeddings",
        map_path=DATA_PATH / "mappings",
        skip_pretraining=args.skip_pretraining,
        randomize_embeddings=args.randomize_embeddings
    )

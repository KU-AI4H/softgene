import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from evaluate import load
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader

def set_all_seeds(seed):
    import random, os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AttentionPoolingGSE(nn.Module):
    def __init__(self, protein_emb_dim, protein_hidden=512, gene_hidden=512):
        super().__init__()
        from torch_scatter import scatter_sum, scatter_softmax
        self.protein_attention_mlp = nn.Sequential(
            nn.Linear(protein_emb_dim, protein_hidden),
            nn.Tanh(),
            nn.Linear(protein_hidden, 1)
        )
        self.gene_attention_mlp = nn.Sequential(
            nn.Linear(protein_emb_dim, gene_hidden),
            nn.Tanh(),
            nn.Linear(gene_hidden, 1)
        )

    def forward(self, protein_embeddings, protein_gene_index, gene_indices_flat, set_indices_flat, num_genes, num_sets):
        from torch_scatter import scatter_sum, scatter_softmax
        attn_p = scatter_softmax(self.protein_attention_mlp(protein_embeddings), protein_gene_index, dim=0)
        gene_embs = scatter_sum(attn_p * protein_embeddings, protein_gene_index, dim=0, dim_size=num_genes)
        
        flat_genes = gene_embs.index_select(0, gene_indices_flat.squeeze(-1))
        attn_g = scatter_softmax(self.gene_attention_mlp(flat_genes), set_indices_flat, dim=0)
        return scatter_sum(attn_g * flat_genes, set_indices_flat, dim=0, dim_size=num_sets)

class SoftPromptLLM(nn.Module):
    def __init__(self, llm_name, soft_prompt_len, protein_emb_dim):
        super().__init__()
        self.encoder = AttentionPoolingGSE(protein_emb_dim=protein_emb_dim)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, device_map="auto", torch_dtype=torch.bfloat16)
        self.llm_dim = self.llm.model.embed_tokens.embedding_dim
        
        self.projector = nn.Sequential(
            nn.Linear(protein_emb_dim, self.llm_dim),
            nn.LayerNorm(self.llm_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.llm_dim, soft_prompt_len * self.llm_dim),
            nn.LayerNorm(soft_prompt_len * self.llm_dim),
            nn.Dropout(0.1),
        )
        self.soft_prompt_len = soft_prompt_len

    def generate(self, protein_data, set_idx, input_ids, attention_mask, max_new_tokens=50):
        all_set_embs = self.encoder(*protein_data)
        batch_set_emb = all_set_embs[set_idx]
        soft_prompts = self.projector(batch_set_emb).view(-1, self.soft_prompt_len, self.llm_dim)

        text_embeds = self.llm.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([soft_prompts, text_embeds], dim=1)
        
        soft_mask = torch.ones((input_ids.size(0), self.soft_prompt_len), device=input_ids.device)
        full_mask = torch.cat([soft_mask, attention_mask], dim=1)

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.llm.config.eos_token_id
        )

def evaluate_model(
    seed,
    model_path,
    data_paths,
    batch_size=4,
    model_id="",
    soft_prompt_length=10,
    out_dir="",
    omit_gpt_context=False,
    randomize_embeddings=False
):
    set_all_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    protein_embeddings = torch.load(data_paths['emb'] / "gene_embedding_stacked.pt")
    protein_gene_index = torch.load(data_paths['map'] / "protein_gene_index.pt").view(-1).long()
    gene_indices_flat = torch.load(data_paths['map'] / "gene_indices_flat.pt").view(-1).long()
    set_indices_flat = torch.load(data_paths['map'] / "set_indices_flat.pt").view(-1).long()
    
    with open(data_paths['map'] / "gene_to_idx.json", 'r') as f: n_genes = len(json.load(f))
    with open(data_paths['map'] / "set_to_idx.json", 'r') as f: set_to_idx = json.load(f)
    
    if randomize_embeddings:
        protein_embeddings = torch.randn_like(protein_embeddings)

    protein_data = [
        d.to(device, dtype=torch.bfloat16) if isinstance(d, torch.Tensor) and torch.is_floating_point(d) 
        else d.to(device, dtype=torch.long) if isinstance(d, torch.Tensor) 
        else d 
        for d in (protein_embeddings, protein_gene_index, gene_indices_flat, set_indices_flat, n_genes, len(set_to_idx))
    ]

    with open(data_paths['gene_sets'], "r") as f: gene_sets = json.load(f)
    with open(data_paths['llm_context'], "r") as f: gpt_responses = json.load(f)
    
    checkpoint = torch.load(model_path / "best_checkpoint.pt", map_location="cpu")

    MODEL_ID = model_id
    SOFT_PROMPT_LEN = soft_prompt_length # Or whatever your value was
    PROTEIN_EMB_DIM = protein_embeddings.shape[-1] # This comes from your loaded embeddings

    model = SoftPromptLLM(
        llm_name=MODEL_ID, 
        soft_prompt_len=SOFT_PROMPT_LEN, 
        protein_emb_dim=PROTEIN_EMB_DIM
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device, dtype=torch.bfloat16).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompt_template = "Input: Genes: {gene_symbols}. {gpt_analysis}\nTask: Predict GO term.\nOutput:"
    
    set_ids = list(set_to_idx.keys())
    _, test_indices = train_test_split(range(len(set_ids)), test_size=0.3, random_state=seed)

    test_prompts, test_targets = [], []
    for idx in test_indices:
        sid = set_ids[idx]
        genes = ', '.join(gene_sets[sid]['genes'][:25])
        gpt_analysis = gpt_responses[sid] if (sid in gpt_responses and not omit_gpt_context) else ""
        test_prompts.append(prompt_template.format(gene_symbols=genes, gpt_analysis=gpt_analysis))
        test_targets.append(gene_sets[sid]['description'])

    print(f"Input dtype: {protein_data[0].dtype}")
    print(f"Projector weight dtype: {model.projector[0].weight.dtype}")
    print(f"LLM embedding dtype: {model.llm.model.embed_tokens.weight.dtype}")

    all_preds = []
    for i in tqdm(range(0, len(test_prompts), batch_size), desc="Generating"):
        batch_prompts = test_prompts[i:i+batch_size]
        batch_set_indices = torch.tensor(test_indices[i:i+batch_size], device=device)
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        
        output_ids = model.generate(protein_data, batch_set_indices, inputs.input_ids, inputs.attention_mask, max_new_tokens=50)
        
        prompt_len = inputs.input_ids.shape[1] + model.soft_prompt_len
        generated_ids = output_ids[:, :]
        
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_preds.extend([d.strip() for d in decoded])

    print("Evaluating Metrics...")
    rouge = load("rouge")
    meteor = load("meteor")
    bertscore = load("bertscore")

    print(f"Computing ROUGE-L score")
    r_res = rouge.compute(predictions=all_preds, references=test_targets)
    
    print("Computing METEOR")
    m_res = meteor.compute(predictions=all_preds, references=test_targets)
    print("Computing BERTScore")
    bs_res = bertscore.compute(predictions=all_preds, references=test_targets, device=device, lang="en", model_type="distilbert-base-uncased")

    print(f"\nSeed {seed} | ROUGE-L: {r_res['rougeL']:.4f} | BERTScore: {np.mean(bs_res['f1']):.4f} | METEOR: {m_res['meteor']:.4f}")

    out_dir = Path(out_dir) / "results.json"
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    results_dict = {"rouge_l": r_res['rougeL'], "bertscore": np.mean(bs_res['f1']), "meteor": m_res['meteor']}

    with open(out_dir, "w") as f:
        json.dump(results_dict, f, indent=2)

    return {"rougeL": r_res['rougeL'], "meteor": m_res['meteor'], "bertscore": np.mean(bs_res['f1'])}

def parse_args():
    parser = ArgumentParser()
    
    # Path Arguments
    parser.add_argument("-m", "--model-path", type=str, required=True,
                        help="Path to the directory containing the saved model checkpoints")
    parser.add_argument("-o", "--output-path", type=str, default="./outputs/eval/",
                        help="Directory where evaluation results and predictions will be saved")
    
    # Evaluation Settings
    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="The random seed to evaluate (should match the training seed)")
    parser.add_argument("-b", "--batch-size", type=int, default=4,
                        help="Batch size for LLM inference (default: 4)")
    parser.add_argument("-n", "--max-samples", type=int, default=None,
                        help="Limit the number of test samples to evaluate (for quick testing)")
    parser.add_argument("-e", "--soft-prompt-length", type=int, default=10)

    # Logic Flags
    parser.add_argument("-c", "--omit-gpt-context", action="store_true",
                        help="If set, the GPT-generated analysis will be removed from the prompt")
    parser.add_argument("-r", "--randomize-embeddings", action="store_true",
                        help="Baseline test: replaces biological embeddings with random noise")
    
    # Data Paths
    parser.add_argument("-l", "--llm-context", default="responses.json",
                        help="Path to the JSON file containing GPT-generated context")
    parser.add_argument("-g", "--data-path", default="./data/",
                        help="Directory containing the protein/gene set embeddings")
    parser.add_argument("-t", "--gene-sets", default="go_gene_sets_all.json",
                        help="Path to the master Gene Ontology gene sets JSON")
    parser.add_argument("-i", "--model-id")

    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    DATA_PATH = Path(args.data_path)

    data_paths = {
        'emb': DATA_PATH / "embeddings",
        'map': DATA_PATH / "mappings",
        'gene_sets': Path(args.gene_sets),
        'llm_context': Path(args.llm_context)
    }
    
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_all_seeds(args.seed)
    
    print(f"🚀 Initializing Evaluation for Seed: {args.seed}")
    print(f"📍 Model Path: {args.model_path}")
    print(f"📊 Batch Size: {args.batch_size}")
    print(f"Output Path: {args.output_path}")
    
    results = evaluate_model(
        seed=args.seed,
        model_path=Path(args.model_path),
        data_paths=data_paths,
        batch_size=args.batch_size,
        model_id=args.model_id,
        soft_prompt_length=args.soft_prompt_length,
        out_dir=Path(args.output_path),
        omit_gpt_context=args.omit_gpt_context,
        randomize_embeddings=args.randomize_embeddings
    )
    
    print("Evaluation Complete. Results saved to:", args.output_path)

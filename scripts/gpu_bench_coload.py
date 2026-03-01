"""Test gpu_layers=-1 with embeddings model co-loaded (simulates server)."""
from llama_cpp import Llama
import time, gc

MODEL = "D:/Offline_Rag_V2/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant. Answer using ONLY the provided context.<|im_end|>\n"
    "<|im_start|>user\nCONTEXT:\n"
    + ("The software provides advanced editing tools for professional workflows. " * 60) +
    "\n\nQUESTION: What features does this software provide?<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# First load sentence-transformers to simulate server's embedding model on GPU
print("Loading embedding model on GPU...")
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("D:/Offline_Rag_V2/models/embeddings/bge-small-en-v1.5", device="cuda")
# Do a test encode to force CUDA alloc
embed_model.encode(["test"])
print("Embedding model loaded on GPU")

import subprocess
res = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
print(f"VRAM after embeddings: {res.stdout.strip()} MiB")

configs = [
    {"layers": -1, "flash": True,  "label": "layers=-1  flash=True"},
    {"layers": -1, "flash": False, "label": "layers=-1  flash=False"},
    {"layers": 33, "flash": True,  "label": "layers=33  flash=True"},
    {"layers": 33, "flash": False, "label": "layers=33  flash=False"},
]

for cfg in configs:
    print(f"\n--- {cfg['label']} ---")
    try:
        m = Llama(
            model_path=MODEL, n_gpu_layers=cfg["layers"], n_ctx=4096,
            n_batch=512, flash_attn=cfg["flash"], n_threads=6, verbose=False,
        )
        res = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        print(f"  VRAM: {res.stdout.strip()} MiB")

        prompt_tokens = len(m.tokenize(PROMPT.encode("utf-8")))
        print(f"  Prompt: {prompt_tokens} tok")

        t0 = time.perf_counter()
        first_tok = None
        count = 0
        for chunk in m.create_completion(
            prompt=PROMPT, max_tokens=200, temperature=0.3,
            stop=["<|im_end|>"], stream=True,
        ):
            if first_tok is None:
                first_tok = time.perf_counter() - t0
            count += 1

        total = time.perf_counter() - t0
        gen_t = total - (first_tok or 0)
        print(f"  TTFT: {first_tok:.2f}s | Gen: {count} tok in {gen_t:.2f}s = {count/gen_t:.1f} tok/s | Total: {total:.2f}s")
    except Exception as e:
        print(f"  CRASHED: {e}")

    try:
        del m
    except:
        pass
    gc.collect()
    time.sleep(2)

del embed_model
gc.collect()

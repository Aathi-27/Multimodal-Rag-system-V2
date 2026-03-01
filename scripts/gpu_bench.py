"""Quick GPU config benchmark for Qwen2.5-1.5B Q4_K_M."""
from llama_cpp import Llama
import time, subprocess, gc

MODEL = "D:/Offline_Rag_V2/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nExplain what a vector database is and how it works in 200 words.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

configs = [
    {"n_ctx": 4096, "n_batch": 512,  "n_threads": 6,  "label": "ctx=4096 batch=512 t=6"},
    {"n_ctx": 2048, "n_batch": 512,  "n_threads": 6,  "label": "ctx=2048 batch=512 t=6"},
    {"n_ctx": 2048, "n_batch": 1024, "n_threads": 6,  "label": "ctx=2048 batch=1024 t=6"},
    {"n_ctx": 2048, "n_batch": 512,  "n_threads": 8,  "label": "ctx=2048 batch=512 t=8"},
    {"n_ctx": 2048, "n_batch": 2048, "n_threads": 6,  "label": "ctx=2048 batch=2048 t=6"},
]

for cfg in configs:
    print(f"\n--- {cfg['label']} ---")
    m = Llama(
        model_path=MODEL,
        n_gpu_layers=-1,
        n_ctx=cfg["n_ctx"],
        n_batch=cfg["n_batch"],
        flash_attn=True,
        n_threads=cfg["n_threads"],
        verbose=False,
    )
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    print(f"  VRAM: {res.stdout.strip()} MiB")

    # Warm up
    m.create_completion(prompt=PROMPT, max_tokens=5, temperature=0.3)

    # Benchmark
    t0 = time.perf_counter()
    out = m.create_completion(prompt=PROMPT, max_tokens=200, temperature=0.3, stop=["<|im_end|>"])
    t1 = time.perf_counter()
    tokens = out["usage"]["completion_tokens"]
    print(f"  {tokens} tok in {t1-t0:.2f}s = {tokens/(t1-t0):.1f} tok/s")

    del m
    gc.collect()
    time.sleep(2)

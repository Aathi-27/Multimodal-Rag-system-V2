"""Test KV cache quantization impact + thread count on generation speed."""
from llama_cpp import Llama, GGML_TYPE_Q8_0, GGML_TYPE_F16
import time, subprocess, gc

MODEL = "D:/Offline_Rag_V2/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# RAG-length prompt (~2000 tokens)
FILLER = "The video editing software provides advanced features for professional content creation. It includes timeline editing, color grading, audio mixing, and effect compositing. " * 50
PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant. Answer using ONLY the provided context. "
    "Cite sources. Be thorough.<|im_end|>\n"
    f"<|im_start|>user\nCONTEXT:\n{FILLER}\n\n"
    "QUESTION: What are the main features of the video editing software?<|im_end|>\n"
    "<|im_start|>assistant\n"
)

configs = [
    {"type_k": None,         "type_v": None,         "n_threads": 6, "label": "KV=f16 t=6 (current)"},
    {"type_k": GGML_TYPE_Q8_0, "type_v": GGML_TYPE_Q8_0, "n_threads": 6, "label": "KV=q8_0 t=6"},
    {"type_k": GGML_TYPE_Q8_0, "type_v": GGML_TYPE_Q8_0, "n_threads": 8, "label": "KV=q8_0 t=8"},
    {"type_k": None,         "type_v": None,         "n_threads": 8, "label": "KV=f16 t=8"},
]

for cfg in configs:
    print(f"\n--- {cfg['label']} ---")
    kwargs = dict(
        model_path=MODEL,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_batch=512,
        flash_attn=True,
        n_threads=cfg["n_threads"],
        verbose=False,
    )
    if cfg["type_k"] is not None:
        kwargs["type_k"] = cfg["type_k"]
        kwargs["type_v"] = cfg["type_v"]

    m = Llama(**kwargs)

    prompt_tokens = len(m.tokenize(PROMPT.encode("utf-8")))
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    print(f"  VRAM: {res.stdout.strip()} MiB | Prompt: {prompt_tokens} tok")

    # Warm up
    m.create_completion(prompt="Hello", max_tokens=5, temperature=0.3)

    # Benchmark stream
    t0 = time.perf_counter()
    first_token_time = None
    count = 0
    for chunk in m.create_completion(
        prompt=PROMPT, max_tokens=300, temperature=0.3,
        stop=["<|im_end|>"], stream=True,
        top_k=1,  # deterministic, less overhead
    ):
        if first_token_time is None:
            first_token_time = time.perf_counter() - t0
        count += 1
    total = time.perf_counter() - t0
    gen_time = total - (first_token_time or 0)

    print(f"  TTFT: {first_token_time:.2f}s ({prompt_tokens/(first_token_time or 1):.0f} tok/s eval)")
    print(f"  Gen:  {count} tok in {gen_time:.2f}s = {count/gen_time:.1f} tok/s")
    print(f"  Total: {total:.2f}s")

    del m
    gc.collect()
    time.sleep(2)

print("\nDone.")

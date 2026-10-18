from transformers import AutoTokenizer
import json
import os
import numpy as np
from FlagEmbedding import FlagModel


# ---------- инструменты ----------
curdir = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(curdir, "RAG_data")

_tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
tokenizer = lambda s: len(_tok.encode(s, add_special_tokens=False))

# query_instruction_for_retrieval — важно для поиска!
model = FlagModel(
    "BAAI/bge-small-en-v1.5",
    use_fp16=True,
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
)

MAX_LEN = 512   # лимит bge-small-en-v1.5


def load_data():
    all_chunks = []
    with open(os.path.join(outdir, "brain_chunks.json"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_chunks.append(json.loads(line))

    emb = np.load(os.path.join(outdir, "embeddings.npy"))
    return all_chunks, emb


def search(all_chunks: list, emb: np.ndarray, query: str, top_k: int = 5):
    q = model.encode_queries([query], max_length=MAX_LEN)[0]
    scores = emb @ q
    idx = np.argsort(-scores)[:top_k]
    return [(all_chunks[i], float(scores[i])) for i in idx]


def main():
    chunks, embs = load_data()
    while True:
        query = input("You: ").strip()
        if query in ["q", "quit", "exit"]:
            print("BYE!!!")
            break
        if not query:
            continue

        raw_k = input("Number of chunks: ").strip()
        try:
            k = int(raw_k) if raw_k else 5
        except ValueError:
            k = 5

        ans = search(all_chunks=chunks, emb=embs, query=query, top_k=k)
        for chunk, score in ans:
            print(f"\n--- score={score:.4f} | {chunk.get('chunk_id', '?')} ---")
            print(chunk["text"][:500])
        print()


if __name__ == "__main__":
    main()
import json
import re
from rank_bm25 import BM25Okapi
import pickle

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from adapters import AutoAdapterModel



    

def prepare_data(filename:str) -> list:
    data = []
    with open(filename, "r", encoding="utf-8") as docs:
        for doc in docs:
            j_doc = json.loads(doc)
            clean_doc = {
                "id": j_doc.get("id", 0),
                "title": j_doc.get("title", ""),
                "abstract": j_doc.get("abstract", "")
            }
            data.append(clean_doc)

    return data

def delete_latex(text:str) ->  str:
    text = re.sub(r"\\(mathbb|mathcal|mathfrak|mathbf|mathrm)\{([A-Za-z])\}", r"\2", text)
    text = re.sub(r"\^\{?([a-zA-Z0-9])\}?", r"\1", text)
    text = re.sub(r"_\{?([a-zA-Z0-9])\}?", r"\1", text)
    text = re.sub(r"\\[a-zA-Z0-9]+", " ", text)
    for ch in "$[{}\\]":
        text.replace(ch, "")
    return text




def bmsearch(bm25, tokens, top_k):
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = {}
    for i in top_idx:
        if scores[i] <= 0:
            continue
        results[i] = scores[i]
    return results


@torch.no_grad()
def neyro_search(model, doc_embs, tokens, top_k):
    result = {}

    ADAPTER2 = "allenai/specter2_adhoc_query"
    model.load_adapter(ADAPTER2, source='hf', set_active=True)
    outputs = model(**tokens)
    query_emb = outputs.last_hidden_state[:, 0, :]
    query_emb = F.normalize(query_emb, p=2, dim=1)
    q_emb = query_emb.numpy()
    
    scores = q_emb @ doc_embs.T
    top_idx = np.argsort(scores[0])[-top_k:][::-1]

    for i in top_idx:
        result[i] = scores[0][i]
    return result


def search(docs, bm25, tokenizer, model, doc_embs, query, top_k):
    results = []

    n_tokens = tokenizer(query, padding=True, truncation=True,
                    return_tensors="pt", return_token_type_ids=False, max_length=512)
    bm_tokens = tokenizer.tokenize(query)
    
    n_res = neyro_search(model, doc_embs, n_tokens, top_k)
    bm_res = bmsearch(bm25, bm_tokens, top_k)

    #множества индексов
    n_i = [i for i in n_res]
    bm_i = [i for i in bm_res]
    inter = set(n_i).intersection(set(bm_i))

    for i in inter:
        res = {
            "type": "neyro + bm",
            "n_score": n_res[i],
            "bm_score": bm_res[i],
            "id": docs[i]["id"],
            "title": docs[i]["title"],
            "abstract": docs[i]["abstract"]
        }
        results.append(res)

    for i in n_res:
        if i in inter:
            continue
        res = {
            "type": "neyro",
            "n_score": n_res[i],
            "id": docs[i]["id"],
            "title": docs[i]["title"],
            "abstract": docs[i]["abstract"]
        }
        results.append(res)

    for i in bm_res:
        if i in inter:
            continue
        res = {
            "type": "bm",
            "bm_score": bm_res[i],
            "id": docs[i]["id"],
            "title": docs[i]["title"],
            "abstract": docs[i]["abstract"]
        }
        results.append(res)


    return results


def main():
    MODEL = "allenai/specter2_base"

    model = AutoAdapterModel.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    #грузим эмбеддинги для нейро поиска
    doc_embs = np.load("technical_data/doc_embs.npy")

    #грузим дату для вывода данных о найденных статьях
    with open("science_data/id_title_abstract_for_olny_mathAG.json", encoding="utf-8") as f:
        data = json.load(f)

    #грузим статистический корпус для статистического поиска
    with open("technical_data/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    


    while True:
        query = input("Ask your question:")
        if query in ["exit", "q", "quit", "end"]:
            break
        top_k = int(input("How much answers you want?:"))

        for r in search(data, bm25, tokenizer, model, doc_embs, query, top_k):
                print("type:", r["type"], end="   ")
                if "n_score" in r: print("n_score:", r["n_score"], end = "   ")
                if "bm_score" in r: print("bm_score:", r["bm_score"], end = "   ")
                print("\n", end="")
                print("id:", r["id"], "   title:", r["title"], "\n", "abstract:", r["abstract"], end="\n\n\n")

# Flexibility of spherical varieties
main()
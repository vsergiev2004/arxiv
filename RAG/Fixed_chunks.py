import os
import pymupdf  # PyMuPDF
from collections import Counter
import re
from transformers import AutoTokenizer
import json
import numpy as np
from FlagEmbedding import BGEM3FlagModel


def extract_blocks(pdf_path):
    doc = pymupdf.open(pdf_path)
    blocks = []
    for page_num, page in enumerate(doc):
        page_height = page.rect.height
        d = page.get_text("dict")   # структурированный вывод
        for block in d["blocks"]:
            if block.get("type") != 0:   # 0 = текст, 1 = картинка
                continue
            for line in block["lines"]:
                text = "".join(span["text"] for span in line["spans"])
                if not text.strip():
                    continue
                # берём параметры первого span'а
                span = line["spans"][0]
                blocks.append({
                    "page": page_num,
                    "text": text.strip(),
                    "x0": line["bbox"][0],
                    "y0": line["bbox"][1],
                    "y1": line["bbox"][3],
                    "page_height": page_height,
                    "font_size": round(span["size"], 1),
                    "font": span["font"],
                    "flags": span["flags"],   # bold/italic биты
                })
    return blocks


def normalize(s):
    # убираем цифры страниц, пробелы — чтобы "3. Methods" и "5. Methods" совпали
    s = re.sub(r"\d+", "#", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def find_running_headers(blocks, top_ratio=0.12, bottom_ratio=0.12, min_pages=0.4):
    """Находит строки, повторяющиеся в верхней/нижней зоне на многих страницах."""
    total_pages = len({b["page"] for b in blocks})
    counter = Counter()

    for b in blocks:
        zone = None
        if b["y0"] < b["page_height"] * top_ratio:
            zone = "top"
        elif b["y1"] > b["page_height"] * (1 - bottom_ratio):
            zone = "bottom"
        else:
            continue
        counter[(zone, normalize(b["text"]))] += 1

    # оставляем те, что встречаются на >= min_pages доле страниц
    threshold = total_pages * min_pages
    running = {key for key, cnt in counter.items() if cnt >= threshold}
    return running


def remove_running_headers(blocks, running):
    cleaned = []
    for b in blocks:
        zone = None
        if b["y0"] < b["page_height"] * 0.12:
            zone = "top"
        elif b["y1"] > b["page_height"] * 0.88:
            zone = "bottom"
        key = (zone, normalize(b["text"]))
        if key in running:
            continue
        cleaned.append(b)
    return cleaned


def make_sent_block(blocks):
    fonts = {}
    max_font = ""

    sent_blocks = []
    
    sent_text = []
    sent_pages = []
    old_font_size = blocks[0]["font_size"]
    x0 = 0
    for b in blocks:
        page = b["page"]
        text = b["text"]
        font_size = b["font_size"]

        if sent_text == []:
            x0 = b["x0"]

        #если сменился шрифт, то и предложение сменилось
        if old_font_size != font_size and sent_text != []:
            if sent_pages == []:
                sent_pages.append(page)
            sentance = {
                "pages": sent_pages[0],
                "text": "".join(sent_text),
                "font_size": old_font_size,
                "x0": x0
                }
            
            sent_blocks.append(sentance)
            sent_pages = []
            sent_text = []
            old_font_size = font_size


        if len(sent_pages) == 0 or page != sent_pages[-1]:
                sent_pages.append(page)
        
        if text[-1] == "-":
            sent_text.append(text[:-1])
        elif text[-1] in ".;:!?□":
            sent_text.append(text)
            
            sentance = {
                "pages": "".join([str(sent_pages[0]), "-", str(sent_pages[-1])]) if len(sent_pages) != 1 else str(sent_pages[0]),
                "text": "".join(sent_text),
                "font_size": font_size,
                "x0": x0
                }

            sent_blocks.append(sentance)
            sent_pages = []
            sent_text = []

            fonts[font_size] = fonts.get(font_size, 0) + 1
            if fonts[font_size] > fonts.get(max_font, 0):
                max_font = font_size

        else:
            sent_text.append(text)
            sent_text.append(" ")


    for sen in sent_blocks:
        if sen["font_size"] == max_font:
            sen["type"] = "regular"
        else:
            sen["type"] = "unregular"

    return sent_blocks


def x0_max_sort(blocks):
    d = x0_counter(blocks)
    m1 = 0
    m2 = 0
    x01 = 0
    x02 = 0
    for x in d:
        if d[x] > m1:
            m2 = m1
            x02 = x01
            m1 = d[x]
            x01 = x
        elif d[x] > m2:
            m2 = d[x]
            x02 = x

    return x01, x02
            

def make_abzats_block(blocks):
    abzats_blocks = []
    struct = ["Lemma", "Theorem", "Deﬁnition", "Proposition", "Example", "Remark", "Proof", "Proof."]
    not_new_abz_x0, new_abz_x0 = x0_max_sort(blocks)

    abz_text = []
    abz_page = None
    abz_type = "nothing spetial"
    for b in blocks:
        page = b["pages"]
        text = b["text"]
        x0 = b["x0"]

        if (x0 != not_new_abz_x0 and abz_type == "nothing spetial") or (abz_type == "Proof." and (abz_text[-1] == "□" or text.split()[0] in struct)) or (abz_type in ["Lemma", "Theorem", "Deﬁnition", "Proposition", "Example", "Remark"] and x0 != new_abz_x0):
            if abz_page is None:
                abz_page = page
            abz = {
                "page": abz_page,
                "text": "".join(abz_text),
                "type": abz_type
            }
            abzats_blocks.append(abz)
            abz_text = [text]
            abz_page = page
            abz_type = "nothing spetial"
        else:
            abz_text.append(text)

        if text != "" and text.split()[0] in ["Lemma", "Theorem", "Deﬁnition", "Proposition", "Example", "Remark", "Proof", "Proof."]:
            abz_type = text.split()[0]

    return abzats_blocks        


STATEMENT_TYPES = {
    "lemma", "theorem", "proposition", "corollary",
    "definition", "remark", "example", "claim", "fact",
}
PROOF_TYPES = {"proof."}

def _norm(t):
    return (t or "nothing").strip().lower()


def build_blocks(items):
    """Склеивает proof с предшествующим statement."""
    blocks = []
    for it in items:
        t = _norm(it.get("type"))
        if t in PROOF_TYPES and blocks and _norm(blocks[-1]["types"][-1]) in STATEMENT_TYPES:
            blocks[-1]["items"].append(it)
            blocks[-1]["types"].append(t)
        else:
            blocks.append({"types": [t], "items": [it]})
    return blocks


def make_chunks_fixed(items, tokenizer, max_tokens=512, overlap_items=1):
    """
    items: [{"page": ..., "text": str, "type": str}, ...] — абзацы.
    Клеит подряд идущие абзацы, пока не превысит max_tokens.
    Абзац не разрезается. overlap_items — сколько последних абзацев
    переносить в следующий чанк (0 = без overlap).
    """
    out = []
    cur = []
    cur_tokens = 0

    def flush():
        nonlocal cur, cur_tokens
        if cur:
            text = "\n\n".join(i["text"].strip() for i in cur if i.get("text", "").strip())
            if text:
                out.append({
                    "text": text,
                    "page": cur[0].get("page", 0),
                    "pages": sorted({i.get("page", 0) for i in cur}),
                    "types": sorted({i.get("type", "nothing spetial") for i in cur}),
                    "n_items": len(cur),
                    "n_tokens": cur_tokens,
                })
        cur, cur_tokens = [], 0

    for it in items:
        text = (it.get("text") or "").strip()
        if not text:
            continue
        t = tokenizer(text)

        # абзац сам по себе больше лимита — отдельным чанком
        if t > max_tokens:
            flush()
            out.append({
                "text": text,
                "page": it.get("page", 0),
                "pages": [it.get("page", 0)],
                "types": [it.get("type", "nothing spetial")],
                "n_items": 1,
                "n_tokens": t,
            })
            continue

        # переполнение — закрываем чанк
        if cur and cur_tokens + t > max_tokens:
            tail = cur[-overlap_items:] if overlap_items else []
            flush()
            cur = list(tail)
            cur_tokens = sum(tokenizer(x["text"]) for x in cur)

        cur.append(it)
        cur_tokens += t

    flush()
    return out



def font_counter(blocks):
    fonts = {}
    for b in blocks:
        fonts[b["font"]] = fonts.get(b["font"], 0) + 1

    return fonts


def font_size_counter(blocks):
    font_sizes = {}
    for b in blocks:
        font_sizes[b["font_size"]] = font_sizes.get(b["font_size"], 0) + 1

    return font_sizes


def x0_counter(blocks):
    x0s = {}
    for b in blocks:
        x = int(b["x0"] + 0.2)
        x0s[x] = x0s.get(x, 0) + 1
    rem = set()
    for x in x0s:
        if x0s[x] == 1:
            rem.add(x)
    for x in rem:
        x0s.pop(x)
    return x0s





def main():
    _tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    tokenizer = lambda s: len(_tok.encode(s, add_special_tokens=False))
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    curdir = os.path.dirname(os.path.abspath(__file__))
    print(curdir)
    datadir = os.path.join(curdir, "RAG_science_data")
    outdir = os.path.join(curdir, "RAG_data")

    all_chunks = []
    

    for name in os.listdir(datadir):
        pdf_path = os.path.join(datadir, name)

        blocks = extract_blocks(pdf_path)
        headers = find_running_headers(blocks) 
        clean_blocks = remove_running_headers(blocks, headers)
        sentance_block = make_sent_block(clean_blocks)
        abzats_block = make_abzats_block(sentance_block)
    
        chunks = make_chunks_fixed(abzats_block, tokenizer=tokenizer, max_tokens=512, overlap_items=1)
        for i, c in enumerate(chunks):
            c["doc_name"] = name
            c["chunk_id"] = f"{name}::{i}"
            all_chunks.append(c)

    with open(os.path.join(outdir, "chunks.json"), "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
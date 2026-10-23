import os
import pymupdf  # PyMuPDF
from collections import Counter
import re
#from transformers import AutoTokenizer
import json
import numpy as np
#from FlagEmbedding import FlagModel
import unicodedata



def delete_latex(text:str) ->  str:
    text = re.sub(r"\\(mathbb|mathcal|mathfrak|mathbf|mathrm)\{([A-Za-z])\}", r"\2", text)
    text = re.sub(r"\^\{?([a-zA-Z0-9])\}?", r"\1", text)
    text = re.sub(r"_\{?([a-zA-Z0-9])\}?", r"\1", text)
    text = re.sub(r"\\[a-zA-Z0-9]+", " ", text)
    for ch in "$[{}\\]":
        text.replace(ch, "")
    return text



def clean_pdf_text(text: str) -> str:

    # 1) выкидываем управляющие символы (\x00 и т.п.), кроме \n и \t
    text = "".join(
        ch for ch in text
        if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )

    # 2) нестандартные пробелы → обычный пробел
    text = re.sub(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]", " ", text)

    # 3) оставляем буквы, цифры, пунктуацию и базовую математику
    keep = set("+=<>~^$±×÷≤≥≠≈∞√∑∏∫∂∆°")
    text = "".join(
        ch if (ch.isspace() or unicodedata.category(ch)[0] in "LNP" or ch in keep)
        else " "
        for ch in text
    )

    # 4) причёсываем пробелы
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_blocks(pdf_path):
    doc = pymupdf.open(pdf_path, filetype="txt")
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
        text = clean_pdf_text(b["text"]) #delete_latex(b["text"])
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
        
        if len(text) == 0:
            continue
        elif text[-1] == "-":
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


def split_by_tokens(text, tokenizer, max_tokens):
    """Режет текст на куски <= max_tokens: сначала по предложениям, затем по словам."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    cur, cur_tok = [], 0
    for s in sents:
        t = tokenizer(s)
        if t > max_tokens:
            # предложение само гигантское — режем по словам
            words = s.split()
            buf = []
            for w in words:
                if buf and tokenizer(" ".join(buf + [w])) > max_tokens:
                    yield " ".join(buf)
                    buf = [w]
                else:
                    buf.append(w)
            if buf:
                yield " ".join(buf)
            continue
        if cur and cur_tok + t > max_tokens:
            yield " ".join(cur)
            cur, cur_tok = [s], t
        else:
            cur.append(s)
            cur_tok += t
    if cur:
        yield " ".join(cur)






def make_chunks_block(items, tokenizer, max_tokens=512, min_tokens=256):
    """
    items: [{"text": str, "type": str, "page": int}, ...] для одного документа.
    tokenizer: Callable[[str], int]
    """
    blocks = build_blocks(items)
    out = []
    cur = []
    cur_tokens = 0
    cur_types = []
    cur_has_statement = False

    def flush():
        nonlocal cur, cur_tokens, cur_types, cur_has_statement
        if cur:
            text = "\n\n".join(i["text"].strip() for i in cur if i.get("text", "").strip())
            if text:
                out.append({
                    "text": text,
                    "types": sorted(set(cur_types)),
                    "page": cur[0].get("page", 0),
                    "n_items": len(cur),
                    "n_tokens": cur_tokens,
                })
        cur, cur_tokens, cur_types, cur_has_statement = [], 0, [], False

    for block in blocks:
        btext = "\n".join(i["text"].strip() for i in block["items"] if i.get("text", "").strip())
        if not btext:
            continue
        btokens = tokenizer(btext)
        btypes = block["types"]
        is_stmt = any(t in STATEMENT_TYPES for t in btypes)

        # блок не влезает целиком — режем его на под-чанки
        if btokens > max_tokens:
            flush()
            buf, buf_tok, buf_types, buf_page = [], 0, [], None
            for it in block["items"]:
                t = (it.get("text") or "").strip()
                if not t:
                    continue
                tk = tokenizer(t)

                # сам item слишком большой — режем по предложениям/словам
                if tk > max_tokens:
                    if buf:
                        out.append({
                            "text": "\n".join(buf),
                            "types": sorted(set(buf_types)),
                            "page": buf_page if buf_page is not None else it.get("page", 0),
                            "n_items": len(buf),
                            "n_tokens": buf_tok,
                        })
                        buf, buf_tok, buf_types, buf_page = [], 0, [], None
                    for piece in split_by_tokens(t, tokenizer, max_tokens):
                        if not piece.strip():
                            continue
                        out.append({
                            "text": piece,
                            "types": sorted(set(btypes)),
                            "page": it.get("page", 0),
                            "n_items": 1,
                            "n_tokens": tokenizer(piece),
                        })
                    continue

                if buf and buf_tok + tk > max_tokens:
                    out.append({
                        "text": "\n".join(buf),
                        "types": sorted(set(buf_types)),
                        "page": buf_page if buf_page is not None else it.get("page", 0),
                        "n_items": len(buf),
                        "n_tokens": buf_tok,
                    })
                    buf, buf_tok, buf_types, buf_page = [], 0, [], None

                buf.append(t)
                buf_tok += tk
                buf_types.extend(btypes)
                if buf_page is None:
                    buf_page = it.get("page", 0)

            if buf:
                out.append({
                    "text": "\n".join(buf),
                    "types": sorted(set(buf_types)),
                    "page": buf_page if buf_page is not None else 0,
                    "n_items": len(buf),
                    "n_tokens": buf_tok,
                })
            continue

        # переполнение
        if cur and cur_tokens + btokens > max_tokens and cur_tokens >= min_tokens:
            flush()

        # не мешаем два утверждения в один чанк
        if is_stmt and cur_has_statement and cur_tokens >= min_tokens:
            flush()

        cur.extend(block["items"])
        cur_tokens += btokens
        cur_types.extend(btypes)
        if is_stmt:
            cur_has_statement = True

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
    #_tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    # поднимаем лимит, чтобы encode() не ругался — используем только для подсчёта токенов
    #_tok.model_max_length = 10**9
    #tokenizer = lambda s: len(_tok.encode(s, add_special_tokens=False))

    '''
    model = BGEM3FlagModel("BAAI/bge-small-en-v1.5", use_fp16=True)
    '''
    curdir = os.path.dirname(os.path.abspath(__file__))
    print(curdir)
    datadir = os.path.join(curdir, "RAG_science_data")
    outdir = os.path.join(curdir, "RAG_data")
    os.makedirs(outdir, exist_ok=True)

    all_chunks = []

    for name in os.listdir(datadir):

        if not name.lower().endswith(".tex") or name != "Affine_toric_emb.tex":
            continue
        tex_path = os.path.join(datadir, name)

        blocks = extract_blocks(tex_path)
        headers = find_running_headers(blocks)
        clean_blocks = remove_running_headers(blocks, headers)
        sentance_block = make_sent_block(clean_blocks)
        abzats_block = make_abzats_block(sentance_block)

        for i in range(0, len(abzats_block)):
            print(abzats_block[i])

    ''' chunks = make_chunks_block(abzats_block, tokenizer=tokenizer)
        for i, c in enumerate(chunks):
            c["doc_name"] = name
            c["chunk_id"] = f"{name}::{i}"
            all_chunks.append(c)

    with open(os.path.join(outdir, "brain_chunks.json"), "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
'''

if __name__ == "__main__":
    main()

'''
    texts = [c["text"] for c in all_chunks]


    emb = model.encode(
        texts,
        batch_size=8,
        max_length=1024,       # совпадает с max_tokens в чанкере
        return_dense=True,
        return_sparse=False,   # sparse/colbert можно позже
        return_colbert_vecs=False,
    )["dense_vecs"]            # (N, 1024), уже нормализованы

    emb = np.asarray(emb, dtype=np.float32)

    

    np.save(outdir / "embeddings.npy", emb)


    # метаданные индекса — чтобы не гадать потом
    meta = {
        "model": "BAAI/bge-m3",
        "dim": int(emb.shape[1]),
        "n_chunks": int(emb.shape[0]),
        "max_length": 1024,
    }
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"saved {len(all_chunks)} chunks, emb shape {emb.shape}")
'''
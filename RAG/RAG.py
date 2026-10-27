import json
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from FlagEmbedding import FlagModel

load_dotenv()

# ─── LLM Client ───────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("GROQ_API_RAG_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "openai/gpt-oss-120b"


curdir = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(curdir, "RAG_data")

MODEL_NAME = "BAAI/bge-m3"  # было BAAI/bge-small-en-v1.5
MAX_LEN = 512
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = FlagModel(
            MODEL_NAME,
            use_fp16=True,
            query_instruction_for_retrieval=QUERY_INSTRUCTION,
        )
    return _model


def load_data():
    chunks = []
    with open(os.path.join(outdir, "brain_chunks.json"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    emb = np.load(os.path.join(outdir, "embeddings.npy"))
    return chunks, emb


def search(chunks, embeddings, query, top_k=5):
    model = _get_model()
    q = model.encode_queries([query], max_length=MAX_LEN)[0]
    scores = embeddings @ q
    idx = np.argsort(-scores)[:top_k]
    return [(chunks[i], float(scores[i])) for i in idx]


def print_results(results):
    for chunk, score in results:
        print(f"\n--- score={score:.4f} | {chunk.get('chunk_id', '?')} ---")
        print(f"Section: {chunk.get('section', '?')}")
        print(f"Types:   {chunk.get('types', '?')}")
        print(f"Text:    {chunk.get('text', '')[:500]}")
        print()


# ─── Данные агента ─────────────────────────────────────────

_chunks = None
_embs = None


def _ensure_loaded():
    global _chunks, _embs
    if _chunks is None or _embs is None:
        _chunks, _embs = load_data()
    return _chunks, _embs


# ─── Тулы ──────────────────────────────────────────────────

def search_chunks(query: str, top_k: int = 5) -> list:
    """Поиск релевантных чанков по текстовому запросу."""
    chunks, embs = _ensure_loaded()
    results = search(chunks, embs, query, top_k=top_k)
    # возвращаем только нужные поля + score
    return [
        {
            "chunk_id": r[0].get("chunk_id"),
            "section": r[0].get("section"),
            "types": r[0].get("types"),
            "score": round(r[1], 4),
            "text": r[0].get("text", "")[:500],
        }
        for r in results
    ]


def get_chunk_detail(chunk_id: str) -> dict:
    """Возвращает полный текст чанка по его ID."""
    chunks, _ = _ensure_loaded()
    for c in chunks:
        if c.get("chunk_id") == chunk_id:
            return {
                "chunk_id": c.get("chunk_id"),
                "section": c.get("section"),
                "types": c.get("types"),
                "text": c.get("text", ""),
            }
    return {"error": f"chunk_id={chunk_id} not found"}


# ─── Описание инструментов для function calling ─────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": (
                "Поиск релевантных чанков текста из базы знаний по запросу. "
                "Возвращает chunk_id, section, types, score и фрагмент текста. "
                "Используй для любого запроса, где нужно найти информацию в документах."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос на естественном языке",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Количество результатов",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk_detail",
            "description": (
                "Получение полного текста чанка по его chunk_id. "
                "Используй, когда нужен полный текст конкретного чанка, "
                "а не только фрагмент."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "ID чанка из результатов search_chunks",
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
]

tool_map = {
    "search_chunks": search_chunks,
    "get_chunk_detail": get_chunk_detail,
}


# ─── Системный промпт ─────────────────────────────────────

SYSTEM_PROMPT = """\
Ты — RAG-агент для работы с базой знаний. Помогаешь находить \
и анализировать информацию из документов.

Твои инструменты:
1. search_chunks(query, top_k) — поиск релевантных чанков по запросу.
   Возвращает chunk_id, section, types, score и фрагмент текста.
2. get_chunk_detail(chunk_id) — полный текст чанка по ID.
   Используй, когда нужен полный контекст конкретного чанка.

Правила:
- По умолчанию используй top_k=5; увеличь, если ответов мало.
- Формулируй поисковые запросы на английском для лучшего качества.
- В ответе указывай chunk_id, section и цитаты из текста.
- Если одного чанка недостаточно — вызывай get_chunk_detail для полного контекста.
- Не выдумывай информацию. Всё, что отдаёшь, должно приходить из инструментов.
"""


# ─── Агентский цикл ───────────────────────────────────────

def run_agent():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 60)
    print("  RAG Knowledge Base Agent")
    print("  Введите 'quit' для выхода")
    print("=" * 60, "\n")

    while True:
        user_input = input("\nВы: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("До встречи!")
            break

        messages.append({"role": "user", "content": user_input})

        # Цикл вызова инструментов (может быть несколько итераций)
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
            )

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                print(f"\nАгент: {msg.content}")
                break

            # Выполняем каждый вызов инструмента
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"  ⚙ {name}({json.dumps(args, ensure_ascii=False)})")

                try:
                    result = tool_map[name](**args)
                except Exception as e:
                    result = {"error": str(e)}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })


if __name__ == "__main__":
    run_agent()

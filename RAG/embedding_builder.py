# embedding_builder.py - Loading chunks and computing embeddings
import os
import json
import torch
import numpy as np
from FlagEmbedding import FlagModel


def compute_embeddings(out_dir, model_name='BAAI/bge-m3'):
    """Вычисляет эмбеддинги для уже сохранённых чанков."""
    curdir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(curdir, out_dir) if not os.path.isabs(out_dir) else out_dir
    
    # Определяем устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Загружаем чанки
    chunks_path = os.path.join(out_dir, 'brain_chunks.json')
    all_chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_chunks.append(json.loads(line))
    
    print(f'Loaded {len(all_chunks)} chunks')
    
    # Загружаем модель на устройство
    print('Loading model...')
    model = FlagModel(
        model_name,
        device=device,
        use_fp16=True,
        query_instruction_for_retrieval='Represent this sentence for searching relevant passages: ',
        model_format='huggingface',
    )
    
    # Эмбеддинги
    print(f'Computing embeddings on {device}...')
    texts = [c['text'] for c in all_chunks]
    embs = model.encode(texts, max_length=512)
    embs = np.asarray(embs, dtype=np.float32)
    
    # Сохранение
    emb_path = os.path.join(out_dir, 'embeddings.npy')
    np.save(emb_path, embs)
    
    print(f'Saved embeddings shape {embs.shape} to {emb_path}')
    return embs


if __name__ == '__main__':
    compute_embeddings('RAG_data')

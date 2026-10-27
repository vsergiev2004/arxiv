# chunker.py - Structural chunking for .tex files
import os
import re
import json
from pylatexenc.latex2text import LatexNodes2Text as LatexEncoder
from transformers import AutoTokenizer


STATEMENT_TYPES = {
    'lemma', 'theorem', 'proposition', 'corollary',
    'definition', 'remark', 'example', 'claim', 'fact',
    'conjecture',
}
PROOF_TYPES = {'proof'}

ENV_TYPES = {
    'lemma': 'lemma', 'theorem': 'theorem', 'definition': 'definition',
    'proposition': 'proposition', 'corollary': 'corollary',
    'remark': 'remark', 'example': 'example', 'proof': 'proof',
    'claim': 'claim', 'fact': 'fact', 'conjecture': 'conjecture',
}

STRUCT_PATTERNS = re.compile(
    r'^\s*(Lemma|Theorem|Definition|Proposition|Corollary|Remark|Example|Proof|Claim|Fact|Conjecture)\s+[\d\w.]*[\.:]',
    re.MULTILINE
)

ENV_BLOCK_RE = re.compile(
    r'\\begin\{(' + '|'.join(ENV_TYPES.keys()) + r')\}.*?\\end\{' + '|'.join(ENV_TYPES.keys()) + r'\}',
    re.DOTALL
)

SECTION_RE = re.compile(r'\\(sub)?section(\*)?\{([^}]+)\}')


def is_statement(t):
    return t in STATEMENT_TYPES


def is_proof(t):
    return t in PROOF_TYPES


def _norm(t):
    return (t or 'nothing').strip().lower()


def extract_macros(preamble: str) -> dict:
    macros = {}
    for m in re.finditer(r'\\newcommand\s+\\([A-Za-z]+)\s*\{(.+?)\}(?!\s*\[)', preamble, re.DOTALL):
        macros['\\' + m.group(1)] = m.group(2)
    for m in re.finditer(r'\\def\\([A-Za-z]+)\s+#?\d?\s*\{(.+?)\}', preamble, re.DOTALL):
        macros['\\' + m.group(1)] = m.group(2)
    return macros


def expand_macros(tex: str, macros: dict) -> str:
    for macro, expansion in sorted(macros.items(), key=lambda x: -len(x[0])):
        tex = re.sub(
            re.escape(macro) + r'(\{([^}]*)\})?',
            lambda m: expansion + (m.group(2) or ''),
            tex
        )
    tex = re.sub(r'\\[A-Za-z]+\{([^}]*)\}', r'\1', tex)
    tex = re.sub(r'\\[A-Za-z]+', '', tex)
    return tex


def post_process_latex(text: str) -> str:
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    text = re.sub(r'\\cite\{[^}]*\}', '', text)
    text = re.sub(r'\\footnote\{[^}]*\}', '', text)
    text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textit\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\emph\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\text\{[^,}]*\{([^}]*)\}\}', r'\1', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def latex_to_text(latex: str, macros: dict = None) -> str:
    if macros:
        latex = expand_macros(latex, macros)
    text = LatexEncoder().latex_to_text(latex)
    text = post_process_latex(text)
    return text


def parse_tex_structure(tex_path: str) -> dict:
    for enc in ['utf-8', 'cp1251', 'latin-1']:
        try:
            with open(tex_path, 'r', encoding=enc) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    doc_start = content.find(r'\begin{document}')
    preamble = content[:doc_start] if doc_start >= 0 else ''
    document = content[doc_start:] if doc_start >= 0 else content
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', document, re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else ''
    title_match = re.search(r'\\title\{(.+?)\}', preamble)
    title = title_match.group(1).strip() if title_match else ''
    sections = []
    current_section = None
    current_subsection = None
    for m in SECTION_RE.finditer(document):
        level = 1 if m.group(1) is None else 2
        name = m.group(3).strip()
        pos = m.start()
        if level == 1:
            if current_section:
                current_section['content'] = document[current_section['_pos']:pos].strip()
            current_section = {
                'name': name, 'level': 1, 'parent': None,
                'subsections': [], 'content': '', '_pos': pos,
            }
            current_subsection = None
            sections.append(current_section)
        elif level == 2 and current_section:
            if current_subsection:
                current_subsection['content'] = document[current_subsection['_pos']:pos].strip()
            current_subsection = {
                'name': name, 'level': 2, 'parent': current_section['name'],
                'content': '', '_pos': pos,
            }
            current_section['subsections'].append(current_subsection)
    if current_section:
        current_section['content'] = document[current_section['_pos']:].strip()
    for s in sections:
        s.pop('_pos', None)
        for ss in s['subsections']:
            ss.pop('_pos', None)
    backmatter = {}
    ack_match = re.search(r'\\begin\{acknowledgement\}(.*?)\\end\{acknowledgement\}', document, re.DOTALL)
    if ack_match:
        backmatter['acknowledgements'] = ack_match.group(1).strip()
    return {
        'title': title, 'abstract': abstract,
        'sections': sections, 'backmatter': backmatter,
    }


def extract_items(content: str, macros: dict = None) -> list:
    """Извлекает items из LaTeX-контента секции."""
    items = []
    found_positions = set()
    
    # Сначала ищем окружения (lemma, theorem, proof...)
    for m in ENV_BLOCK_RE.finditer(content):
        env_name = re.search(r'\\begin\{(\w+)\}', m.group(0))
        if env_name:
            env_type = env_name.group(1).lower()
            latex = m.group(0)
            text = latex_to_text(latex, macros) if macros else post_process_latex(latex)
            items.append({'latex': latex, 'text': text, 'type': env_type})
            for i in range(m.start(), m.end()):
                found_positions.add(i)
    
    # Проверяем структуру — ищем "Lemma 1.", "Theorem 2.3" и т.д.
    struct_matches = list(STRUCT_PATTERNS.finditer(content))
    for m in struct_matches:
        start = m.start()
        if start not in found_positions:
            first_word = m.group(1).lower()
            if first_word in ENV_TYPES:
                # Находим конец абзаца
                end_match = re.search(r'\n\s*\n', content[m.end():])
                end = m.end() + end_match.start() if end_match else m.end() + 200
                latex = content[m.start():end].strip()
                text = latex_to_text(latex, macros) if macros else post_process_latex(latex)
                items.append({'latex': latex, 'text': text, 'type': first_word})
                for i in range(m.start(), end):
                    found_positions.add(i)
    
    # Затем находим обычный текст (между окружениями и структурами)
    all_starts = sorted(found_positions)
    last_end = 0
    for pos in sorted(found_positions):
        if pos > last_end:
            before = content[last_end:pos].strip()
            if before:
                latex = before
                text = latex_to_text(latex, macros) if macros else post_process_latex(latex)
                items.append({'latex': latex, 'text': text, 'type': 'regular'})
        last_end = pos + 10  # пропуск
    
    # Остаток после последнего элемента
    after = content[last_end:].strip()
    if after:
        latex = after
        text = latex_to_text(latex, macros) if macros else post_process_latex(latex)
        items.append({'latex': latex, 'text': text, 'type': 'regular'})
    
    return items


def make_structured_chunks(items, section_info, tokenizer, max_tokens=512, min_tokens=200, overlap_items=1):
    """Создаёт чанки из items с учётом section context и overlap."""
    out = []
    cur = []
    cur_tokens = 0
    cur_has_statement = False
    
    def flush(overlap=False):
        nonlocal cur, cur_tokens, cur_has_statement
        if cur:
            out.append({
                'latex': '\n\n'.join(i['latex'] for i in cur),
                'text': '\n\n'.join(i['text'] for i in cur),
                'section': section_info['name'],
                'section_level': section_info['level'],
                'types': sorted({i['type'] for i in cur}),
                'n_tokens': cur_tokens,
                'overlap': overlap,
            })
        cur, cur_tokens, cur_has_statement = [], 0, False
    
    for item in items:
        t = tokenizer(item['text'])
        item_type = item['type']
        
        # item сам > max_tokens -> режем
        if t > max_tokens:
            flush()
            # Режем текст по предложениям
            sents = re.split(r'(?<=[.!?])\s+', item['text'])
            buf, buf_tok = [], 0
            for s in sents:
                st = tokenizer(s)
                if buf and buf_tok + st > max_tokens:
                    out.append({
                        'latex': '\n\n'.join(buf),
                        'text': '\n\n'.join(buf),
                        'section': section_info['name'],
                        'section_level': section_info['level'],
                        'types': [item_type],
                        'n_tokens': buf_tok,
                        'overlap': False,
                    })
                    buf, buf_tok = [], 0
                buf.append(s)
                buf_tok += st
            if buf:
                out.append({
                    'latex': '\n\n'.join(buf),
                    'text': '\n\n'.join(buf),
                    'section': section_info['name'],
                    'section_level': section_info['level'],
                    'types': [item_type],
                    'n_tokens': buf_tok,
                    'overlap': False,
                })
            continue
        
        # переполнение -> flush
        if cur and cur_tokens + t > max_tokens and cur_tokens >= min_tokens:
            flush()
            # overlap: переносим последний item
            tail = cur[-overlap_items:] if overlap_items else []
            cur = list(tail)
            cur_tokens = sum(tokenizer(i['text']) for i in cur)
            cur_has_statement = any(is_statement(i['type']) for i in cur)
        
        # два statement рядом -> не мешать
        if is_statement(item_type) and cur_has_statement and cur_tokens >= min_tokens:
            flush()
        
        cur.append(item)
        cur_tokens += t
        if is_statement(item_type):
            cur_has_statement = True
    
    flush()
    return out


def build_index(tex_dir, out_dir, model_name='BAAI/bge-small-en-v1.5'):
    """Полный пайплайн: парсинг -> чанкинг."""
    curdir = os.path.dirname(os.path.abspath(__file__))
    tex_dir = os.path.join(curdir, tex_dir) if not os.path.isabs(tex_dir) else tex_dir
    out_dir = os.path.join(curdir, out_dir) if not os.path.isabs(out_dir) else out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Загружаем токенизатор
    tok = AutoTokenizer.from_pretrained(model_name)
    tokenizer = lambda s: len(tok.encode(s, add_special_tokens=False))
    
    all_chunks = []
    
    for tex_file in sorted(os.listdir(tex_dir)):
        if not tex_file.endswith('.tex'):
            continue
        
        print(f'Processing {tex_file}...')
        tex_path = os.path.join(tex_dir, tex_file)
        structure = parse_tex_structure(tex_path)
        
        # Извлекаем макросы
        for enc in ['utf-8', 'cp1251', 'latin-1']:
            try:
                with open(tex_path, 'r', encoding=enc) as f:
                    full_text = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
                full_text = f.read()
        doc_start = full_text.find(r'\begin{document}')
        preamble = full_text[:doc_start] if doc_start >= 0 else ''
        macros = extract_macros(preamble)
        
        # Abstract -> отдельный чанк
        if structure['abstract']:
            all_chunks.append({
                'chunk_id': f'{tex_file}::abstract',
                'doc_name': tex_file,
                'latex': structure['abstract'],
                'text': latex_to_text(structure['abstract'], macros),
                'section': 'Abstract',
                'section_level': 0,
                'types': ['abstract'],
                'n_tokens': tokenizer(structure['abstract']),
                'overlap': False,
            })
        
        # Секции -> чанки
        for section in structure['sections']:
            # Content секции (до первой подсекции)
            if section['content']:
                items = extract_items(section['content'], macros)
                chunks = make_structured_chunks(items, section, tokenizer)
                for i, chunk in enumerate(chunks):
                    chunk['chunk_id'] = f'{tex_file}::{section["name"]}::{i}'
                    chunk['doc_name'] = tex_file
                all_chunks.extend(chunks)
            
            # Подсекции рекурсивно
            for subsection in section['subsections']:
                if subsection['content']:
                    items = extract_items(subsection['content'], macros)
                    chunks = make_structured_chunks(items, subsection, tokenizer)
                    for i, chunk in enumerate(chunks):
                        chunk['chunk_id'] = f'{tex_file}::{subsection["name"]}::{i}'
                        chunk['doc_name'] = tex_file
                    all_chunks.extend(chunks)
    
    # Сохранение чанков
    out_path = os.path.join(out_dir, 'brain_chunks.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    
    print(f'Saved {len(all_chunks)} chunks to {out_path}')
    return all_chunks


if __name__ == '__main__':
    build_index('RAG_science_data', 'RAG_data')

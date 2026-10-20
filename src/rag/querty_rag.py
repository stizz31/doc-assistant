import os
import pickle
from pathlib import Path
import faiss
import requests

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = (BASE_DIR / ".." / ".." / "data" / "vector_db").resolve()
FAISS_INDEX_PATH = DB_DIR / "faiss.index"
DOCS_PICKLE_PATH = DB_DIR / "documents_mapping.pkl"

def load_rag_assets():
    """Загружает FAISS индекс и текстовую базу в память один раз при старте сервера."""
    if not FAISS_INDEX_PATH.exists() or not DOCS_PICKLE_PATH.exists():
        raise FileNotFoundError("Ошибка: Файлы базы FAISS не найдены. Сначала запустите build_index.py")
    
    # Легковесный SentenceTransformer загружаем прямо здесь
    from sentence_transformers import SentenceTransformer
    print("Загрузка индекса FAISS и модели эмбеддингов...")
    
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(DOCS_PICKLE_PATH, "rb") as f:
        db_data = pickle.load(f)
    embedder = SentenceTransformer("DeepPavlov/rubert-base-cased-sentence")
    
    # Возвращаем словарь со всеми тяжелыми объектами
    return {"index": index, "db_data": db_data, "embedder": embedder}

def rag_answer(question: str, state: dict, k: int = 3):
    """Ищет контекст в FAISS и генерирует ответ через API Ollama."""
    # 1. Получаем эмбеддинг вопроса и ищем в FAISS
    query_vector = state["embedder"].encode([question], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    
    scores, indices = state["index"].search(query_vector, k)
    
    # 2. Собираем найденный контекст и метаданные
    search_results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        search_results.append({
            "text": state["db_data"]["chunks"][idx],
            "distance": float(score),
            "meta": state["db_data"]["metadatas"][idx]
        })
    
    context = "\n---\n".join([res["text"] for res in search_results])
    
    # 3. Формируем строгий системный промпт под требования лабораторной
    system_prompt = (
        "Ты - строгий робот-помощник. Твоя задача - ответить на вопрос, основываясь ТОЛЬКО на предоставленных фрагментах текста.\n"
        "Если в предоставленном тексте нет точного ответа на вопрос, ты обязан ответить строго фразой: 'В документах не найдено'.\n"
        "Запрещено использовать внешние знания, придумывать факты или дописывать информацию, которой нет в тексте."
    )
    user_prompt = f"ФРАГМЕНТЫ ТЕКСТА:\n{context}\n\nВОПРОС: {question}\nОТВЕТ:"
    
    # 4. Отправляем обычный POST-запрос в Ollama (совместимый с OpenAI формат)
    ollama_url = "http://127.0.0.1:11434/v1/chat/completions"
    payload = {
        "model": "qwen2.5:3b",
        "temperature": 0.1,  # Низкая температура для строгости ответов
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    
    response = requests.post(ollama_url, json=payload)
    response.raise_for_status()
    
    llm_response = response.json()["choices"][0]["message"]["content"].strip()
    return llm_response, search_results



File: doc-assistant/src/app.py
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.rag.query_rag import load_rag_assets, rag_answer

# Сюда запишем тяжелые ресурсы при старте
index_state = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index_state
    print("=== Загрузка RAG-системы ===")
    index_state = load_rag_assets()  # Строим/загружаем индекс один раз при старте сервера
    print("=== Система успешно загружена и готова к запросам ===")
    yield
    print("=== Остановка сервера ===")

app = FastAPI(title="Doc Assistant", lifespan=lifespan)

# Pydantic-схема: описывает форму запроса, FastAPI сам её валидирует и документирует
class Query(BaseModel):
    question: str = Field(..., description="Вопрос к базе документов")
    k: int = Field(default=3, ge=1, le=10, description="Количество фрагментов для поиска")

@app.get("/health")
def health():
    """Эндпоинт для проверки здоровья сервиса"""
    return {"status": "ok"}

@app.post("/ask")
def ask(q: Query):
    """Точка входа для RAG вопросов"""
    # Обязательная обвязка для обработки ошибок (пустой вопрос)
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")
    
    # Замеряем время ответа, чтобы проанализировать производительность CPU
    start_time = time.time()
    
    answer, sources = rag_answer(q.question, index_state, k=q.k)
    
    elapsed_time = time.time() - start_time
    print(f"Запрос обработан за {elapsed_time:.4f} сек.")  # Логирование в консоль
    
    return {
        "answer": answer, 
        "sources": sources,
        "metrics": {"execution_time_seconds": round(elapsed_time, 2)}
    }

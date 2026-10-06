import json
from llm_client import ask_gigachat

def analyze_review_promt(review_text:str) -> str:

    promt = f"""
Ты — профессиональный ИИ-аналитик стримингового каталога Netflix. Твоя задача — анализировать описания фильмов и сериалов.
Определи жанр (одно слово на английском, например: drama, comedy, thriller, documentary, action, romance, horror, sci-fi), 
настроение (mood — одно-два слова на английском, например: dark, uplifting, tense, heartwarming) 
и подходит ли контент для семейного просмотра (family_friendly: true или false).

Выведи результат СТРОГО в формате JSON. Не пиши никаких вводных слов, markdown-разметки или пояснений.

### ПРИМЕРЫ:
Входное описание: "A young woman discovers she has the power to communicate with the dead, and must use her gift to solve a series of mysterious murders in her small town."
Выходной JSON:
{{"genre": "thriller", "mood": "tense", "family_friendly": false}}

Входное описание: "A group of adorable talking animals embarks on a magical adventure to save their forest home from a greedy developer."
Выходной JSON:
{{"genre": "animation", "mood": "heartwarming", "family_friendly": true}}

Входное описание: "A stand-up comedian reflects on his life, career, and the absurdities of modern society in this raw and personal special."
Выходной JSON:
{{"genre": "documentary", "mood": "humorous", "family_friendly": false}}

### РЕАЛЬНОЕ ЗАДАНИЕ:
Входное описание: "{description_text}"
Выходной JSON:
"""
    return promt

if __name__ == "__main__":
    user_review == """ 
    After a tragic accident, a detective becomes obsessed with solving a cold case 
    that has haunted his career for over a decade, risking everything in the process.
    """

    print ("1. Формируем сложный пром...")
    final_promt = analyze_review_promt(user_review)

    print ("2. Отправляем запрос в Gigachat...")
    raw_response = ask_gigachat(final_promt, temperature =0.1)

    print(f"\nСырой ответ модели:\n{raw_response}\n")

    print("3.Проверяем валдиность полученного JSON...")
    try:
        parsed_json = json.loads(raw_response.strip())
        print("УСПЕХ! Данные успешно преобразованы в Python dict:")
        print(f"Жанр: {parsed_json.get('genre')}")
        print(f"Настроение: {parsed_json.get('mood')}")
        print(f"Для семейного просмотра: {parsed_json.get('family_friendly')}")
    except json.JSONDecodeError:
        print("Ошибка: Модель нарушила формат и вернула невалидный JSON.")
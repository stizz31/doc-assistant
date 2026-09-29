import json
from llm_client import ask_gigachat

def analyze_review_promt(review_text:str) -> str:

    promt = f"""
Ты - профессиональный ИИ-аналитик маркетплейсов. Твоя задача - анализировать отзывы клиентов.
Определи тональность отзыва и извлеки главный плюс и гланвый минус

Выведи резулутат СТРОГО в формате JSON. НЕпиши никаких вводных слов, разметки или пояснений"""

    return promt

if __name__ == "__main__":
    user_review == """ 
    """

    print ("1. Формируем сложный пром...")
    final_promt = analyze_review_promt(user_review)

    print ("2. Отправляем запрос в Gigachat...")
    raw_response = ask_gigachat(final_promt, temperature =0.1)

    print(f"\nСырой ответ модели:\n{raw_response}\n")

    print("3.Проверяем валдиность полученного JSON...")
    try:
        parsed_json = json.loads(raw_response.strip())
        print("УСПЕх!Данные успешно преобразованы в Python dict:")
        print(f"тональность: {parsed_json.get('sentiment')}")
        print(f"Плюсы: {parsed_json.get('pros')}")
        print(f"Минусы: {parsed_json.get('cons')}")
    except json.JSONDecodeError:
        print("Ошибка : Модель нарушила формат и вернула невалиднывй JSON.")

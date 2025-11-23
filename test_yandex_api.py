#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы с YandexGPT API через OpenAI-совместимый интерфейс.

Usage:
  export YANDEX_API_KEY=your_api_key
  python test_yandex_api.py
"""

import os
import sys
import time
from openai import OpenAI
import openai

def test_yandex_api():
    """Отправляет тестовый запрос в YandexGPT API."""

    # Проверяем наличие API ключа
    api_key = os.getenv('YANDEX_API_KEY')
    if not api_key:
        print("❌ Ошибка: переменная окружения YANDEX_API_KEY не задана!")
        print("Установите ключ:")
        print("  export YANDEX_API_KEY=your_api_key")
        sys.exit(1)

    # Настройки из конфига
    base_url = "http://llm.iaaras.lan:4000/v1"
    model = "yandex/GPT-OSS-20B"

    print(f"🔗 Подключение к: {base_url}")
    print(f"🤖 Модель: {model}")
    print(f"🔑 API ключ: {api_key[:8]}...{api_key[-4:]}")
    print()

    # Создаем клиент
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # Тестовый промпт
    system_prompt = "Ты помощник для проверки работы API. Отвечай кратко и по делу."
    user_message = "Привет! Это тестовое сообщение. Просто ответь 'API работает корректно' если ты меня понял."

    print("📤 Отправка тестового запроса...")
    print(f"System: {system_prompt[:50]}...")
    print(f"User: {user_message}")
    print()

    # Retry логика для обработки 429 ошибок
    max_retries = 3
    retry_delay = 5  # секунд

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"🔄 Попытка {attempt + 1}/{max_retries}...")
                print()

            # Отправляем запрос
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2
            )

            # Получаем ответ
            answer = response.choices[0].message.content.strip()

            print("✅ Запрос выполнен успешно!")
            print()
            print("📥 Ответ от модели:")
            print("-" * 60)
            print(answer)
            print("-" * 60)
            print()

            # Статистика
            if hasattr(response, 'usage') and response.usage:
                print("📊 Статистика токенов:")
                print(f"  Входные токены:  {response.usage.prompt_tokens}")
                print(f"  Выходные токены: {response.usage.completion_tokens}")
                print(f"  Всего токенов:   {response.usage.total_tokens}")

                # Расчет стоимости (0.1₽ за 1K токенов)
                input_cost = (response.usage.prompt_tokens / 1000) * 0.1
                output_cost = (response.usage.completion_tokens / 1000) * 0.1
                total_cost = input_cost + output_cost
                print(f"  Стоимость:       {total_cost:.4f}₽")
            else:
                print("ℹ️  Информация о токенах недоступна")

            print()
            print("🎉 Тест завершен успешно! YandexGPT API работает корректно.")
            return  # Успех - выходим

        except openai.RateLimitError as e:
            # 429 ошибка - все деплойменты заняты
            if attempt < max_retries - 1:
                print(f"⚠️  Модель занята (429 - все деплойменты используются)")
                print(f"   Повторная попытка через {retry_delay} секунд...")
                print()
                time.sleep(retry_delay)
            else:
                print(f"❌ Модель недоступна после {max_retries} попыток")
                print(f"   {e}")
                print()
                print("Попробуйте:")
                print("  - Подождать несколько минут и запустить снова")
                print("  - Уменьшить нагрузку на сервер")
                sys.exit(1)

        except Exception as e:
            print(f"❌ Ошибка при выполнении запроса:")
            print(f"   {type(e).__name__}: {e}")
            print()
            print("Возможные причины:")
            print("  - Неверный API ключ")
            print("  - Недоступен сервер (проверьте URL)")
            print("  - Проблемы с сетью")
            print("  - Неверное имя модели")
            sys.exit(1)

if __name__ == "__main__":
    test_yandex_api()

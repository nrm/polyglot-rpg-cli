#!/usr/bin/env python3
"""
Тест для проверки нового поведения кеша с глоссарием.

Проверяет, что при изменении глоссария кеш автоматически инвалидируется
для chunks с измененными терминами, но сохраняется для неизмененных chunks.
"""

import tempfile
import json
from pathlib import Path
from polyglot_rpg.main import TranslationCache, Glossary


def test_cache_with_glossary_changes():
    """
    Тест проверяет, что:
    1. Кеш работает по обработанному тексту (с примененным глоссарием)
    2. При изменении глоссария кеш автоматически инвалидируется
    3. Chunks без измененных терминов остаются в кеше
    """
    print("🧪 Запуск теста кеша с глоссарием...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Создаем тестовый кеш
        cache = TranslationCache(tmppath)

        # Создаем глоссарий v1
        glossary_path = tmppath / "glossary_v1.yaml"
        glossary_path.write_text("""
- term: "wizard"
  translation: "маг"
- term: "dragon"
  translation: "дракон"
""")
        glossary_v1 = Glossary(glossary_path)

        # Исходный текст
        text1 = "The wizard attacks"
        text2 = "The knight defends"

        # Применяем глоссарий v1
        processed_text1_v1 = glossary_v1.apply_to_text(text1)
        processed_text2_v1 = glossary_v1.apply_to_text(text2)

        print(f"\n📝 Оригинальные тексты:")
        print(f"  text1: '{text1}'")
        print(f"  text2: '{text2}'")
        print(f"\n✨ После применения глоссария v1 (wizard -> маг):")
        print(f"  text1: '{processed_text1_v1}'")
        print(f"  text2: '{processed_text2_v1}'")

        # Сохраняем "переводы" в кеш
        cache.set(processed_text1_v1, "Маг атакует")
        cache.set(processed_text2_v1, "Рыцарь защищается")
        cache.save()

        # Проверяем, что кеш работает
        assert cache.get(processed_text1_v1) == "Маг атакует", "❌ Кеш не работает для text1 v1"
        assert cache.get(processed_text2_v1) == "Рыцарь защищается", "❌ Кеш не работает для text2 v1"
        print("\n✅ Кеш работает для обработанных текстов v1")

        # Создаем глоссарий v2 с измененным термином
        glossary_path.write_text("""
- term: "wizard"
  translation: "волшебник"
- term: "dragon"
  translation: "дракон"
""")
        glossary_v2 = Glossary(glossary_path)

        # Применяем глоссарий v2
        processed_text1_v2 = glossary_v2.apply_to_text(text1)
        processed_text2_v2 = glossary_v2.apply_to_text(text2)

        print(f"\n✨ После применения глоссария v2 (wizard -> волшебник):")
        print(f"  text1: '{processed_text1_v2}'")
        print(f"  text2: '{processed_text2_v2}'")

        # Проверяем, что для text1 кеш НЕ работает (термин изменился)
        cached_text1_v2 = cache.get(processed_text1_v2)
        assert cached_text1_v2 is None, f"❌ Кеш должен быть инвалидирован для text1 v2, но вернул: {cached_text1_v2}"
        print(f"\n✅ Кеш корректно инвалидирован для text1 (термин 'wizard' изменился)")

        # Проверяем, что для text2 кеш РАБОТАЕТ (термины не изменились)
        cached_text2_v2 = cache.get(processed_text2_v2)
        assert cached_text2_v2 == "Рыцарь защищается", f"❌ Кеш должен работать для text2 v2, но вернул: {cached_text2_v2}"
        print(f"✅ Кеш сохранился для text2 (термины не изменились)")

        print("\n🎉 Все тесты прошли успешно!")
        print("\n📊 Резюме:")
        print("   ✅ Кеш работает по обработанному тексту (с примененным глоссарием)")
        print("   ✅ При изменении глоссария кеш автоматически инвалидируется для затронутых chunks")
        print("   ✅ Chunks без измененных терминов остаются в кеше")


if __name__ == "__main__":
    test_cache_with_glossary_changes()

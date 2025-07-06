#!/usr/bin/env python3
# polyglot_rpg/main.py

import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm
from openai import OpenAI
from markdown_it import MarkdownIt
from tqdm import tqdm
import re
from pathlib import Path
import yaml
import json
from markdown_it.token import Token
import tiktoken
import hashlib
import importlib.resources  # Для доступа к файлам данных внутри пакета
from typing import List, Dict, Optional, Any

from markdownify import markdownify as md_from_html

app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)
console = Console()

# --- Константы для структуры проекта ---
CONFIG_NAME = "01_config.yaml"
INPUT_DIR_NAME = "02_input_chapters"
WORKSPACE_DIR_NAME = "03_translation_workspace"
AST_DIR_NAME = "1_asts"
GLOSSARY_REVIEW_NAME = "2_glossary.for_review.yaml"
GLOSSARY_FINAL_NAME = "2_glossary.final.yaml"
TRANSLATED_AST_DIR_NAME = "3_translated_asts"
FINAL_DIR_NAME = "4_final_chapters"
CACHE_DIR_NAME = ".cache"
DEFAULT_CONFIG_TEMPLATE_NAME = "default_config_template.yaml"

# --- Классы для организации логики ---

class Project:
    """Инкапсулирует пути и конфигурацию проекта."""
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir.resolve()
        
        # Основные директории
        self.input_dir = self.project_dir / INPUT_DIR_NAME
        self.workspace_dir = self.project_dir / WORKSPACE_DIR_NAME
        self.ast_dir = self.workspace_dir / AST_DIR_NAME
        self.translated_ast_dir = self.workspace_dir / TRANSLATED_AST_DIR_NAME
        self.final_dir = self.workspace_dir / FINAL_DIR_NAME
        self.cache_dir = self.workspace_dir / CACHE_DIR_NAME
        
        # Файлы
        self.config_path = self.project_dir / CONFIG_NAME
        self.glossary_review_path = self.workspace_dir / GLOSSARY_REVIEW_NAME
        self.glossary_final_path = self.workspace_dir / GLOSSARY_FINAL_NAME
        
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            console.print(f"[bold red]Ошибка: Конфигурационный файл не найден: {self.config_path}[/bold red]")
            raise typer.Exit(1)
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_source_files(self) -> List[Path]:
        return sorted(list(self.input_dir.glob("*.md")))

class TokenCounter:
    """Простой класс для подсчета токенов и оценки стоимости."""
    def __init__(self, model_name: str = "gpt-4o"):
        self.input_tokens = 0
        self.output_tokens = 0
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.input_cost_per_m = 5.0
        self.output_cost_per_m = 15.0

    def _count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def add_input(self, text: str):
        self.input_tokens += self._count(text)

    def add_output(self, text: str):
        self.output_tokens += self._count(text)

    def report(self, command_name: str):
        total_input_cost = (self.input_tokens / 1_000_000) * self.input_cost_per_m
        total_output_cost = (self.output_tokens / 1_000_000) * self.output_cost_per_m
        total_cost = total_input_cost + total_output_cost
        
        console.print("\n[bold]📊 Статистика использования токенов[/bold]")
        console.print(f"Команда: [cyan]{command_name}[/cyan]")
        console.print(f"   Токены на вход (input):  [green]{self.input_tokens:,}[/green]")
        console.print(f"   Токены на выход (output): [green]{self.output_tokens:,}[/green]")
        console.print(f"   [bold]Итого токенов:[/bold]          [bold green]{(self.input_tokens + self.output_tokens):,}[/bold green]")
        console.print(f"   [yellow]Примерная стоимость (GPT-4o):[/yellow] [bold yellow]${total_cost:.4f}[/bold yellow]")

class TranslationCache:
    """Управляет кэшированием переводов для экономии API вызовов."""
    def __init__(self, cache_dir: Path):
        self.cache_path = cache_dir / "translations_cache.json"
        self.cache_dir = cache_dir
        self.cache = self._load()

    def _load(self) -> Dict[str, str]:
        self.cache_dir.mkdir(exist_ok=True)
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _get_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[str]:
        return self.cache.get(self._get_hash(text))

    def set(self, original_text: str, translated_text: str):
        self.cache[self._get_hash(original_text)] = translated_text

    def save(self):
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

class Glossary:
    """Загружает и предоставляет доступ к терминам из глоссария."""
    def __init__(self, path: Path):
        self.path = path
        self.terms = self._load()

    def _load(self) -> Dict[str, str]:
        if not self.path.exists():
            console.print("📖 Глоссарий не найден. Перевод будет выполнен без него.")
            return {}
        
        console.print(f"📖 Найден глоссарий: [green]{self.path}[/green]")
        with open(self.path, 'r', encoding='utf-8') as f:
            glossary_data = yaml.safe_load(f)
        
        if not glossary_data:
            console.print(f"   [yellow]Предупреждение: Глоссарий пуст.[/yellow]")
            return {}
            
        terms = {item['term']: item['translation'] for item in glossary_data if item.get('term') and item.get('translation')}
        if terms:
            console.print(f"   ✅ Загружено {len(terms)} терминов.")
        else:
            console.print(f"   [yellow]Предупреждение: Глоссарий не содержит валидных записей.[/yellow]")
        return terms

    def apply_to_text(self, text: str) -> str:
        """Применяет глоссарий к тексту, заменяя термины."""
        if not self.terms:
            return text
        
        # Сортируем термины от длинных к коротким, чтобы избежать частичных замен (например, "Mage Hand" перед "Mage")
        for term, translation in sorted(self.terms.items(), key=lambda i: len(i[0]), reverse=True):
             # Используем regex для замены целых слов, чтобы не заменять 'cat' в 'caterpillar'
            text = re.sub(r'\b' + re.escape(term) + r'\b', translation, text, flags=re.IGNORECASE)
        return text

class Translator:
    """Управляет процессом перевода с помощью LLM, кэша и глоссария."""
    def __init__(self, project: Project):
        self.project = project
        api_conf = project.config['api']
        self.client = OpenAI(base_url=api_conf['url'], api_key=api_conf['key'])
        self.model = api_conf['model']
        self.temperature = api_conf.get('temperature', 0.2)
        
        self.token_counter = TokenCounter(model_name=self.model)
        self.cache = TranslationCache(project.cache_dir)
        self.glossary = Glossary(project.glossary_final_path)
        
        # Загружаем системный промпт из конфига
        try:
            self.system_prompt = project.config['translation_settings']['system_prompt']
        except KeyError:
            console.print("[bold red]Ошибка: 'system_prompt' не найден в 'translation_settings' в файле конфигурации.[/bold red]")
            raise typer.Exit(1)

        self.failure_phrases = ("please provide the english text", "i'm ready when you are", "i need the english text")

    def translate_chunk(self, chunk: str) -> str:
        """Переводит один фрагмент текста, используя кэш и глоссарий."""
        cached = self.cache.get(chunk)
        if cached:
            return cached

        # Применяем глоссарий к тексту перед отправкой в LLM
        text_to_translate = self.glossary.apply_to_text(chunk)
        
        # Оборачиваем текст в теги <data>
        user_message_content = f"<data>\n{text_to_translate}\n</data>"

        try:
            self.token_counter.add_input(self.system_prompt)
            self.token_counter.add_input(user_message_content)
            
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message_content}
                ],
                temperature=self.temperature,
            )
            raw_translation = resp.choices[0].message.content
            self.token_counter.add_output(raw_translation)

            if any(phrase in raw_translation.lower() for phrase in self.failure_phrases):
                console.print(f"\n[bold yellow]Предупреждение: модель отказалась переводить. Используется оригинал.[/bold yellow]")
                translation = chunk # Возвращаем оригинальный чанк, не пред-обработанный
            else:
                translation = re.sub(r"<think>.*?</think>", "", raw_translation, flags=re.DOTALL).strip()
            
            self.cache.set(chunk, translation)
            return translation

        except Exception as e:
            console.print(f"\n[yellow]Предупреждение: не удалось перевести чанк '{chunk[:30]}...'. ({e})[/yellow]")
            return chunk # Возвращаем оригинал в случае ошибки

    def save_cache(self):
        self.cache.save()

# --- Вспомогательные функции ---

def _tokens_to_json(tokens: List[Token]) -> List[Dict]:
    """Сериализует токены в JSON-совместимый формат."""
    return [t.as_dict() for t in tokens]

def _extract_strings_from_json(data) -> list[str]:
    """Рекурсивно извлекает все строковые значения из любой вложенной структуры."""
    found_strings = []
    if isinstance(data, dict):
        for value in data.values():
            found_strings.extend(_extract_strings_from_json(value))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                found_strings.append(item)
            else:
                found_strings.extend(_extract_strings_from_json(item))
    return found_strings

# --- Команды CLI ---

@app.command()
def init(project_dir: Path = typer.Argument(..., help="Директория для создания нового проекта.")):
    """Инициализирует новую структуру проекта для перевода."""
    console.print(f"▶️  Инициализация проекта в [cyan]{project_dir}[/cyan]...")
    project_dir.mkdir(exist_ok=True)
    (project_dir / INPUT_DIR_NAME).mkdir(exist_ok=True)
    workspace = project_dir / WORKSPACE_DIR_NAME
    workspace.mkdir(exist_ok=True)
    (workspace / AST_DIR_NAME).mkdir(exist_ok=True)
    (workspace / TRANSLATED_AST_DIR_NAME).mkdir(exist_ok=True)
    (workspace / FINAL_DIR_NAME).mkdir(exist_ok=True)
    (workspace / CACHE_DIR_NAME).mkdir(exist_ok=True)
    (project_dir / INPUT_DIR_NAME / ".gitkeep").touch()
    
    config_path = project_dir / CONFIG_NAME
    if not config_path.exists():
        # Читаем шаблон из ресурсов пакета и копируем его в проект пользователя.
        try:
            # Находим путь к файлу шаблона внутри нашего пакета ('polyglot_rpg')
            template_path = importlib.resources.files('polyglot_rpg').joinpath(DEFAULT_CONFIG_TEMPLATE_NAME)
            # Читаем содержимое шаблона
            template_content = template_path.read_text(encoding='utf-8')
            # Записываем содержимое в новый конфигурационный файл
            config_path.write_text(template_content, encoding='utf-8')
            console.print(f"📄 Создан конфигурационный файл: [green]{config_path}[/green]")
        except FileNotFoundError:
            console.print(f"[bold red]Критическая ошибка: Файл шаблона '{DEFAULT_CONFIG_TEMPLATE_NAME}' не найден внутри пакета.[/bold red]")
            raise typer.Exit(1)
    else:
        console.print(f"📄 Конфигурационный файл уже существует: [yellow]{config_path}[/yellow]")
    
    console.print(f"\n[bold green]✅ Проект '{project_dir.name}' успешно инициализирован![/bold green]")
    console.print(f"➡️  Теперь поместите ваши исходные .md файлы в директорию [cyan]{project_dir / INPUT_DIR_NAME}[/cyan]")


@app.command()
def create_glossary(
    project_dir: Path = typer.Argument(..., help="Путь к директории проекта.", exists=True),
    use_llm: bool = typer.Option(False, "--use-llm", help="Использовать LLM для извлечения и фильтрации терминов."),
    pre_translate: bool = typer.Option(False, "--pre-translate", help="Использовать LLM для предварительного перевода терминов (требует --use-llm)."),
):
    """Сканирует исходные тексты и создает черновик глоссария."""
    if pre_translate and not use_llm:
        console.print("[bold red]Ошибка: Опция --pre-translate может использоваться только вместе с --use-llm.[/bold red]")
        raise typer.Exit(1)
        
    console.print(f"▶️  Запуск создания глоссария для проекта [cyan]{project_dir.name}[/cyan]")
    
    project = Project(project_dir)
    glossary_path = project.glossary_review_path
    
    source_files = project.get_source_files()
    if not source_files:
        console.print(f"[bold yellow]Предупреждение: не найдено .md файлов в {project.input_dir}[/bold yellow]")
        raise typer.Exit()

    all_terms = set()
    translations_map = {}

    if use_llm:
        console.print("🤖 [bold]Режим LLM активирован.[/bold] Это может занять некоторое время.")
        try:
            api_conf = project.config['api']
            glossary_conf = project.config['glossary_settings']
            extraction_prompt = glossary_conf['extraction_prompt']
            filtering_prompt = glossary_conf['filtering_prompt']
            translation_prompt = glossary_conf['translation_prompt']
            client = OpenAI(base_url=api_conf['url'], api_key=api_conf['key'])
            token_counter = TokenCounter(model_name=api_conf['model'])
        except KeyError as e:
            console.print(f"[bold red]Ошибка в файле конфигурации: отсутствует ключ или раздел: {e}.[/bold red]")
            console.print("[bold red]Убедитесь, что ваш config.yaml содержит разделы 'api' и 'glossary_settings' со всеми промптами.[/bold red]")
            raise typer.Exit(code=1)

        # Этап 1: Извлечение
        console.print("🧠 [Этап 1/3] Извлечение терминов с помощью LLM...")
        for source_path in tqdm(source_files, desc="Анализ глав"):
            source_text = source_path.read_text(encoding='utf-8')
            if not source_text.strip(): continue
            try:
                user_content = f"<data>\n{source_text}\n</data>"
                token_counter.add_input(extraction_prompt)
                token_counter.add_input(user_content)
                
                resp = client.chat.completions.create(
                    model=api_conf['model'], messages=[{"role": "system", "content": extraction_prompt}, {"role": "user", "content": user_content}],
                    temperature=0.0, response_format={"type": "json_object"},
                )
                response_content = resp.choices[0].message.content
                token_counter.add_output(response_content)
                
                content = json.loads(response_content)
                found_terms = _extract_strings_from_json(content)
                all_terms.update(found_terms)
            except Exception as e:
                console.print(f"\n[yellow]Предупреждение: ошибка при извлечении из {source_path.name}. ({e})[/yellow]")

        # Этап 2: Фильтрация
        console.print(f"\n🔍 [Этап 2/3] Найдено {len(all_terms)} уник. терминов. Фильтрация...")
        if all_terms:
            try:
                term_list_json = json.dumps({"terms": sorted(list(all_terms))})
                user_content = f"<data>\n{term_list_json}\n</data>"
                token_counter.add_input(filtering_prompt)
                token_counter.add_input(user_content)

                resp = client.chat.completions.create(
                    model=api_conf['model'], messages=[{"role": "system", "content": filtering_prompt}, {"role": "user", "content": user_content}],
                    temperature=0.0, response_format={"type": "json_object"},
                )
                response_content = resp.choices[0].message.content
                token_counter.add_output(response_content)

                content = json.loads(response_content)
                final_terms = _extract_strings_from_json(content)
                all_terms = set(final_terms)
            except Exception as e:
                console.print(f"\n[yellow]Предупреждение: ошибка при фильтрации. ({e})[/yellow]")
        
        # Этап 3: Предварительный перевод
        if pre_translate and all_terms:
            console.print(f"\n🇷🇺 [Этап 3/3] Найдено {len(all_terms)} терминов. Предварительный перевод...")
            try:
                term_list_json = json.dumps(sorted(list(all_terms)))
                user_content = f"<data>\n{term_list_json}\n</data>"
                token_counter.add_input(translation_prompt)
                token_counter.add_input(user_content)

                resp = client.chat.completions.create(
                    model=api_conf['model'], messages=[{"role": "system", "content": translation_prompt}, {"role": "user", "content": user_content}],
                    temperature=0.1, response_format={"type": "json_object"},
                )
                response_content = resp.choices[0].message.content
                token_counter.add_output(response_content)

                translations_map = json.loads(response_content)
                if not isinstance(translations_map, dict):
                    console.print(f"\n[yellow]Предупреждение: LLM вернула некорректный формат для переводов (не словарь).[/yellow]")
                    translations_map = {}
            except Exception as e:
                console.print(f"\n[yellow]Предупреждение: ошибка при предварительном переводе. ({e})[/yellow]")
        
        token_counter.report("Создание глоссария")

    else: # Режим Regex
        console.print("⚡ [bold]Режим Regex активирован (быстрый локальный анализ).[/bold]")
        emphasis_regex = re.compile(r'\*\*(.*?)\*\*|\*(.*?)\*')
        proper_noun_regex = re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b')
        single_cap_word_regex = re.compile(r'(?<!\.\s)\b[A-Z][a-zA-Z\']{3,}\b')
        temp_terms = set()
        for source_path in tqdm(source_files, desc="Анализ глав"):
            source_text = source_path.read_text(encoding='utf-8')
            for line in source_text.splitlines():
                for regex in [emphasis_regex, proper_noun_regex, single_cap_word_regex]:
                    for result in regex.findall(line):
                        term = result if isinstance(result, str) else next((s for s in result if s), None)
                        if term and len(term) > 2 and term.lower() not in ["the", "and", "for", "with"]:
                            temp_terms.add(term.strip())
        all_terms = temp_terms

    if not all_terms:
        console.print("[bold yellow]Не найдено потенциальных терминов для глоссария.[/bold yellow]")
        raise typer.Exit()

    console.print(f"\n📝 Найдено {len(all_terms)} терминов. Поиск контекста...")
    potential_terms_with_context = {}
    full_text = "\n".join([p.read_text(encoding='utf-8') for p in source_files])
    
    for term in tqdm(sorted(list(all_terms)), desc="Поиск контекста"):
        context_regex = re.compile(r'([^\n]*\b' + re.escape(term) + r'\b[^\n]*)', re.IGNORECASE)
        match = context_regex.search(full_text)
        context = match.group(1).strip() if match else "Контекст не найден."
        potential_terms_with_context[term] = (context[:200] + '...') if len(context) > 200 else context

    glossary_for_review = [
        {"term": term, "translation": translations_map.get(term, ""), "context": context}
        for term, context in sorted(potential_terms_with_context.items())
    ]
    header = ("# Этот файл сгенерирован автоматически...\n")
    with open(glossary_path, 'w', encoding='utf-8') as f:
        f.write(header)
        yaml.dump(glossary_for_review, f, allow_unicode=True, sort_keys=False, width=120)

    console.print(f"\n[bold green]✅ Черновик глоссария создан![/bold green]")
    console.print(f"📄 Файл сохранен в: [green]{glossary_path}[/green]")
    console.print("➡️  [bold]Ваши действия:[/bold] Откройте файл, проверьте переводы и сохраните как '2_glossary.final.yaml'.")


# Вспомогательная функция для сборки Markdown из inline токенов
def build_markdown_from_inline(tokens: List[Token]) -> str:
    """Собирает строку Markdown из списка inline токенов и их дочерних элементов."""
    content = ""
    for token in tokens:
        if token.type == 'text':
            content += token.content
        elif token.type == 'strong_open':
            content += '**'
        elif token.type == 'strong_close':
            content += '**'
        elif token.type == 'em_open':
            content += '*'
        elif token.type == 'em_close':
            content += '*'
        elif token.type == 's_open':
            content += '~~'
        elif token.type == 's_close':
            content += '~~'
        # Можно добавить обработку других inline-элементов, если нужно
    return content

@app.command()
def translate(project_dir: Path = typer.Argument(..., help="Путь к директории проекта.", exists=True)):
    """Запускает полный конвейер перевода для проекта."""
    console.print(f"▶️  Запуск конвейера перевода для проекта [cyan]{project_dir.name}[/cyan]")
    
    project = Project(project_dir)
    translator = Translator(project)
    
    all_source_files = project.get_source_files()
    if not all_source_files:
        console.print(f"[bold yellow]Предупреждение: не найдено .md файлов в {project.input_dir}[/bold yellow]")
        raise typer.Exit()

    # --- Интерактивный выбор файлов ---
    console.print("\n[bold]Выберите файлы для перевода:[/bold]")
    for i, file_path in enumerate(all_source_files):
        console.print(f"  [cyan]{i + 1}[/cyan]: {file_path.name}")
    console.print("  [cyan]all[/cyan]: Перевести все файлы")
    
    choice = Prompt.ask("Введите номера файлов через запятую (например, 1,3,5) или 'all'", default="all")
    
    files_to_process = []
    if choice.lower() == 'all':
        files_to_process = all_source_files
    else:
        try:
            indices = [int(i.strip()) - 1 for i in choice.split(',')]
            files_to_process = [all_source_files[i] for i in indices if 0 <= i < len(all_source_files)]
        except (ValueError, IndexError):
            console.print("[bold red]Ошибка: Неверный ввод. Пожалуйста, введите корректные числа или 'all'.[/bold red]")
            raise typer.Exit(1)
            
    if not files_to_process:
        console.print("[yellow]Не выбрано ни одного файла. Завершение работы.[/yellow]")
        raise typer.Exit()
        
    console.print("\n[green]Будут обработаны следующие файлы:[/green]")
    for file_path in files_to_process:
        console.print(f"  - {file_path.name}")

    if not Confirm.ask("\nПродолжить?", default=True):
        raise typer.Exit()

    # --- Основной цикл перевода ---
    md = MarkdownIt("commonmark").enable(["table", "strikethrough"])

    for source_path in tqdm(files_to_process, desc="Обработка глав"):
        chapter_name = source_path.stem
        console.print(f"\n[bold]--- Обработка главы: {chapter_name} ---[/bold]")
        
        source_text = source_path.read_text(encoding='utf-8')
        tokens = md.parse(source_text)
        
        ast_path = project.ast_dir / f"{chapter_name}.ast.json"
        with open(ast_path, 'w', encoding='utf-8') as f:
            json.dump(_tokens_to_json(tokens), f, indent=2, ensure_ascii=False)
        console.print(f"💾 Оригинальный AST сохранен в [green]{ast_path}[/green]")

        console.print("🧠 Перевод блоков...")
        
        # --- Основной цикл обработки токенов с прогресс-баром ---
        with tqdm(total=len(tokens), desc=f"Перевод {chapter_name}", leave=False) as pbar:
            i = 0
            while i < len(tokens):
                token = tokens[i]
                pbar.update(1)
                
                parent_type = tokens[i-1].type if i > 0 else ''
                
                is_translatable_content = (
                    token.type == 'inline' and
                    parent_type in [
                        'paragraph_open', 'heading_open', 'list_item_open',
                        'th_open', 'td_open'
                    ]
                )

                if is_translatable_content:
                    text_to_translate = build_markdown_from_inline(token.children)
                    
                    if text_to_translate.strip():
                        translated_text = translator.translate_chunk(text_to_translate)
                        parsed_inline_tokens = md.parseInline(translated_text.strip(), {})
                        
                        if parsed_inline_tokens and parsed_inline_tokens[0].children:
                            token.children = parsed_inline_tokens[0].children
                            token.content = translated_text
                        else:
                            text_token = Token('text', '', 0)
                            text_token.content = translated_text
                            token.children = [text_token]
                            token.content = translated_text

                elif token.type == 'fence':
                    original_content = token.content.strip()
                    if original_content:
                        translated_content = translator.translate_chunk(original_content)
                        token.content = translated_content + '\n'
                
                i += 1

        translated_ast_path = project.translated_ast_dir / f"{chapter_name}.translated.ast.json"
        with open(translated_ast_path, 'w', encoding='utf-8') as f:
            json.dump(_tokens_to_json(tokens), f, indent=2, ensure_ascii=False)
        console.print(f"💾 Переведенный AST сохранен в [green]{translated_ast_path}[/green]")

        console.print("📝 Рендеринг в финальный Markdown...")
        md_out = md.renderer.render(tokens, md.options, {})
        final_md = md_from_html(md_out, heading_style="ATX")
        final_md = re.sub(r"\n{3,}", "\n\n", final_md)
        
        final_path = project.final_dir / f"{source_path.name}"
        final_path.write_text(final_md, encoding='utf-8')
        console.print(f"✅ Глава сохранена в [bold green]{final_path}[/bold green]")

    translator.save_cache()
    console.print("\n💾 Кэш переводов сохранен.")
    console.print("\n[bold green]🎉 Выбранные главы успешно обработаны! Перевод завершен.[/bold green]")
    translator.token_counter.report("Перевод глав")

if __name__ == "__main__":
    app()
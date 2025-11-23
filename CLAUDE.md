# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Polyglot-RPG** is a Python CLI tool for semi-automated translation of large text materials (TTRPG rulebooks) using LLMs. It parses Markdown files into AST, extracts terminology, and leverages LLM-assisted translation while preserving formatting and allowing user control over critical terminology.

The core philosophy is **"Control > Full Automation"** — the tool automates routine tasks (file parsing, API requests, text assembly) but keeps quality decisions under user control.

## Architecture & Code Structure

The entire application is contained in `/polyglot_rpg/main.py` (~750+ lines) with no additional modules. The architecture is monolithic but well-organized into logical sections:

### Core Data Models (Classes)

1. **Project** (`main.py:39-67`)
   - Encapsulates project structure and paths
   - Manages directory layout: `02_input_chapters/`, `03_translation_workspace/` with subdirectories
   - Loads and provides access to configuration from `01_config.yaml`
   - Method `get_source_files()` retrieves all `.md` files from input directory

2. **TokenCounter** (`main.py:69-100`)
   - Tracks input/output token usage for cost estimation
   - Uses `tiktoken` library to count tokens for the specified model
   - Provides `report()` method for displaying statistics after operations

3. **TranslationCache** (`main.py:102-130`)
   - Manages caching of translations to reduce API calls
   - Stores cache as JSON at `.cache/translations_cache.json`
   - Uses SHA256 hashing of text chunks as keys
   - Load/save operations handle missing files gracefully

4. **Glossary** (`main.py:132-167`)
   - Loads YAML glossary file (either `2_glossary.final.yaml` or `2_glossary.for_review.yaml`)
   - `apply_to_text()` method uses regex with word boundaries to replace terms
   - Sorts terms longest-first to prevent partial replacements (e.g., "Mage" inside "Mage Hand")

5. **Translator** (`main.py:169-232`)
   - Orchestrates the translation process using LLM, cache, and glossary
   - Initializes OpenAI client for any OpenAI-compatible API (Ollama, LM Studio, etc.)
   - `translate_chunk()` method:
     - Checks cache first
     - Applies glossary terms
     - Sends to LLM with system prompt
     - Detects failure phrases and falls back to original text
     - Removes `<think>` tags (for extended thinking models)

### CLI Commands

Four main commands are exposed via Typer framework (`app = typer.Typer()`):

1. **`init <project_dir>`**
   - Creates standardized project structure
   - Creates directories: `02_input_chapters/`, `03_translation_workspace/1_asts/`, `3_translated_asts/`, `4_final_chapters/`, `.cache/`
   - Copies config template from package resources to `01_config.yaml`

2. **`create-glossary <project_dir> [OPTIONS]`**
   - Three-stage LLM-driven glossary creation (if `--use-llm` enabled):
     - **Stage 1**: Extract key terms from all input files using extraction prompt
     - **Stage 2**: Filter extracted terms using filtering prompt
     - **Stage 3**: Pre-translate terms (if `--pre-translate` enabled) using translation prompt
   - Without LLM: performs simple regex-based term extraction
   - Outputs `2_glossary.for_review.yaml` for user review and correction
   - User must save corrected version as `2_glossary.final.yaml`

3. **`translate <project_dir>`**
   - Main translation pipeline with interactive file selection
   - Parses each markdown file into AST using `markdown-it-py`
   - Identifies translatable text nodes (paragraph, inline, list items)
   - For each translatable chunk:
     - Applies glossary terms BEFORE checking cache
     - Translates via `Translator.translate_chunk()`
     - Updates AST with translated content
   - Reconstructs markdown from modified AST
   - Outputs translated files to `4_final_chapters/`
   - Saves translation cache for future runs

4. **`proofread <project_dir>`**
   - Post-translation editing for grammar, agreement, and style
   - Interactive file selection (like translate)
   - Reads translated files from `4_final_chapters/`
   - Smart text splitting by tokens (not paragraphs):
     - Uses `_split_text_by_tokens()` function
     - Auto-calculates block size: 30% of `context_length` (configurable)
     - Example: 16384 context → ~4915 tokens per block
   - For each block:
     - Checks proofreading cache first
     - Finds relevant glossary terms in the block (not all 100+ terms)
     - Sends to LLM with prompt: fix grammar/style, preserve meaning and terms
     - Caches results separately from translation cache
   - Works in-place (overwrites `4_final_chapters/`)
   - Git integration: warns about uncommitted changes before overwriting

### Helper Functions

- **`_tokens_to_json()`** (`main.py:236-238`): Serializes markdown-it Token objects to JSON-compatible dicts
- **`_extract_strings_from_json()`** (`main.py:240-252`): Recursively extracts all string values from nested JSON structures (used for LLM response parsing)
- **`build_markdown_from_inline()`** (`main.py:445-464`): Reconstructs markdown syntax from inline token sequences

## Build & Development Commands

```bash
# Install in editable mode (required for development)
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .

# Install with dev dependencies (pytest, coverage, etc.)
uv sync --extra dev
# Or: pip install -e ".[dev]"

# Run commands
polyglot-rpg init example_project/test_project
polyglot-rpg create-glossary example_project/test_project --use-llm --pre-translate
polyglot-rpg translate example_project/test_project
polyglot-rpg proofread example_project/test_project

# Check installation
which polyglot-rpg
polyglot-rpg --help

# Run tests
pytest tests/
pytest tests/ -v  # verbose output
pytest tests/ --cov=polyglot_rpg  # with coverage
```

## Configuration

The config file (`01_config.yaml`) contains:

- **api**: Settings for main translation API (e.g., local Ollama)
  - `url`: OpenAI-compatible API endpoint
  - `key`: API key (for Ollama usually 'ollama')
  - `model`: Model name (e.g., 'gemma3:27b')
  - `temperature`: Creativity level (0.0-0.2 for accurate translations)
  - `context_length`: Model's context window in tokens (e.g., 16384 for Ollama)

- **proofreading_api**: Settings for proofreading API (can use different model with larger context)
  - `url`: API endpoint (e.g., 'http://llm.iaaras.lan:4000/v1')
  - `key_env_var`: Name of environment variable containing API key (e.g., 'YANDEX_API_KEY')
  - `model`: Model name (e.g., 'yandex/GPT-OSS-20B')
  - `temperature`: Temperature for proofreading (0.1-0.3 recommended)
  - `context_length`: Context window in tokens (e.g., 128000 for YandexGPT)
  - `price_per_1k_input_tokens`: Price in RUB per 1000 input tokens (e.g., 0.1)
  - `price_per_1k_output_tokens`: Price in RUB per 1000 output tokens (e.g., 0.1)

- **translation_settings**: System prompt for main translation
- **glossary_settings**: Three prompts for extraction, filtering, and pre-translation stages
- **proofreading_settings**:
  - `system_prompt`: Instructions for grammar/style checking
  - `max_tokens_per_block`: Block size for proofreading (default: auto = 30% of context_length)

Template is in `/polyglot_rpg/default_config_template.yaml`. Default setup uses Ollama at `http://localhost:11434/v1` with `gemma3:27b` for translation, and YandexGPT with 128K context for proofreading.

### Setting up API keys for proofreading

The proofreading command uses a separate API that requires authentication via environment variable:

```bash
# Linux/Mac - add to ~/.bashrc or ~/.zshrc
export YANDEX_API_KEY=your_api_key_here

# Or set temporarily for one session
export YANDEX_API_KEY=your_api_key_here
polyglot-rpg proofread my_project

# Windows (PowerShell)
$env:YANDEX_API_KEY="your_api_key_here"

# Windows (CMD) - permanent
setx YANDEX_API_KEY "your_api_key_here"
```

If the environment variable is not set, the `proofread` command will display an error with setup instructions.

## Key Implementation Details

### AST-Based Markdown Parsing

The tool parses Markdown into AST tokens rather than simple regex. This preserves formatting perfectly:
- Extracts all text content from relevant token types (paragraph, list items, inline text)
- Translates text chunks independently
- Reconstructs markdown by updating token content and rebuilding from AST

### Translation Strategy

1. **Glossary Pre-processing**: Glossary terms substituted BEFORE cache check (cache is glossary-aware)
2. **Caching**: Translation cache uses SHA256 hash of glossary-processed text as key
   - Changing glossary automatically invalidates affected chunks
   - Unaffected chunks remain cached
3. **Error Handling**: If LLM response contains failure phrases, returns original text
4. **Cost Tracking**: All API calls tracked for token and cost reporting

### Proofreading Strategy

1. **Token-based Splitting**: Text split into blocks by token count (not paragraph count)
   - Uses `tiktoken` for accurate token counting
   - Block size: auto = 30% of `context_length` (configurable)
   - Preserves paragraph boundaries (never splits mid-paragraph)
2. **Selective Glossary**: Only sends terms found in current block (not entire glossary)
3. **Grammatical Gender Support**: For terms with `gender` field, adds separate section in prompt
   - Helps LLM ensure proper agreement with adjectives and verbs
   - Only terms with specified gender are included in this section
4. **Separate Cache**: Proofreading cache independent from translation cache
5. **In-place Updates**: Overwrites `4_final_chapters/` with git safety checks

### Glossary Format with Grammatical Gender

The glossary supports an optional `gender` field (m/f/n) for terms requiring proper grammatical agreement:

```yaml
- term: Basilisk
  translation: Василиск
  gender: m

- term: Mage Hand
  translation: Рука мага
  gender: f

- term: Black Iron
  translation: Черное железо
  gender: n

- term: Alliance
  translation: Союз
  # gender field is optional - omit if not needed
```

**When and how to use:**
- Add `gender` field manually when editing `2_glossary.for_review.yaml`
- Use for proper nouns and terms where agreement matters (typically m/f/n in Russian)
- During `proofread`, terms with gender are passed to LLM in a separate section:
  ```
  Глоссарий терминов (НЕ ИЗМЕНЯТЬ):
  - Basilisk → Василиск
  - Mage Hand → Рука мага

  Грамматические роды для согласования:
  - Василиск (m)
  - Рука мага (f)

  ВАЖНО: Слова с указанным родом должны быть согласованы в тексте...
  ```
- The `translate` command does NOT use gender information (only applies glossary substitution)
- Backward compatible: old glossaries without `gender` continue to work

### User Control Points

- **Glossary review**: Must explicitly rename `2_glossary.for_review.yaml` → `2_glossary.final.yaml` to proceed
- **File selection**: Interactive prompt to choose which chapters to translate/proofread
- **Iterative refinement**: Both caches prevent re-processing, allowing incremental improvements
- **Git integration**: Proofread warns about uncommitted changes before overwriting

## Dependencies

- **typer**: CLI framework
- **openai**: OpenAI-compatible API client
- **markdown-it-py**: Markdown parser
- **markdownify**: HTML to Markdown converter (used for glossary processing)
- **pyyaml**: Config parsing
- **tqdm**: Progress bars
- **tiktoken**: Token counting

All dependencies defined in `pyproject.toml:[project]dependencies`.

## Testing & Running Example

A pre-configured example project exists at `example_project/ironsworn/`:
- Input files in `02_input_chapters/` (3 demo chapters from Ironsworn)
- Pre-generated ASTs in `03_translation_workspace/1_asts/`
- Pre-translated ASTs in `03_translation_workspace/3_translated_asts/`
- Final translated markdown in `03_translation_workspace/4_final_chapters/`
- Example config at project root

To test with this example:
```bash
polyglot-rpg create-glossary example_project/ironsworn --use-llm --pre-translate
# Then review and save glossary
polyglot-rpg translate example_project/ironsworn
```

## Git-Based Workflow (Recommended)

The tool is designed to work seamlessly with git version control. **Recommended workflow:**

### 1. After Translation
```bash
polyglot-rpg translate my_project
git add 03_translation_workspace/4_final_chapters/
git commit -m "translate: Initial translation with glossary"
```

### 2. After Proofreading
```bash
polyglot-rpg proofread my_project
# Tool will warn if there are uncommitted changes in 4_final_chapters/

git diff  # Review what proofreading changed
git add 03_translation_workspace/4_final_chapters/
git commit -m "proofread: Grammar and style fixes"
```

### 3. Iterative Glossary Updates
```bash
# Found a new term while reviewing proofreading results
nano 03_translation_workspace/2_glossary.final.yaml
git commit -am "glossary: Add term 'Ironlands'"

# Re-translate (only affected chunks will be re-processed, thanks to cache)
polyglot-rpg translate my_project
git commit -am "translate: Apply new glossary term"

# Re-proofread (cache speeds this up too)
polyglot-rpg proofread my_project
git diff  # Check changes
git commit -am "proofread: Re-check after glossary update"
```

### 4. Rollback Bad Proofreading
```bash
# If proofreading made unwanted changes:
git revert HEAD
# or
git reset --hard HEAD~1
```

### Key Benefits of Git Integration

- **Safety**: `proofread` command checks for uncommitted changes and warns before overwriting
- **Transparency**: `git diff` shows exactly what proofreading changed
- **Rollback**: Easy to revert bad proofreading results
- **History**: Full audit trail of translation iterations

## Important Notes

- **Python 3.9+** required (uses f-strings, type hints with generics like `list[str]`, Path)
- **Testing**: Tests are located in `tests/` directory. Run with `pytest tests/`. Dev dependencies include pytest and pytest-cov for coverage reporting
- **Error recovery**: Code is defensive with try-except blocks around LLM calls; original text returned on failure
- **Multiline tokens**: The AST reconstruction handles complex nested markdown (tables, nested lists, etc.)
- **Language support**: Currently hardcoded for English→Russian in prompts, but easily configurable via `01_config.yaml`
- **Translation cache**: Cache is glossary-aware - it stores translations based on glossary-processed text, automatically invalidating when glossary terms change
- **Proofreading**: The `proofread` command works in-place (overwrites 4_final_chapters/). Use git commits before/after for safety and diff visibility

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

Three main commands are exposed via Typer framework (`app = typer.Typer()`):

1. **`init <project_dir>`** (`main.py:256-288`)
   - Creates standardized project structure
   - Creates directories: `02_input_chapters/`, `03_translation_workspace/1_asts/`, `3_translated_asts/`, `4_final_chapters/`, `.cache/`
   - Copies config template from package resources to `01_config.yaml`

2. **`create-glossary <project_dir> [OPTIONS]`** (`main.py:292-443`)
   - Three-stage LLM-driven glossary creation (if `--use-llm` enabled):
     - **Stage 1**: Extract key terms from all input files using extraction prompt
     - **Stage 2**: Filter extracted terms using filtering prompt
     - **Stage 3**: Pre-translate terms (if `--pre-translate` enabled) using translation prompt
   - Without LLM: performs simple regex-based term extraction
   - Outputs `2_glossary.for_review.yaml` for user review and correction
   - User must save corrected version as `2_glossary.final.yaml`

3. **`translate <project_dir>`** (`main.py:467-750+`)
   - Main translation pipeline
   - Parses each markdown file into AST using `markdown-it-py`
   - Identifies translatable text nodes (paragraph, inline, list items)
   - For each translatable chunk:
     - Translates via `Translator.translate_chunk()`
     - Updates AST with translated content
   - Reconstructs markdown from modified AST
   - Outputs translated files to `4_final_chapters/`
   - Saves translation cache for future runs

### Helper Functions

- **`_tokens_to_json()`** (`main.py:236-238`): Serializes markdown-it Token objects to JSON-compatible dicts
- **`_extract_strings_from_json()`** (`main.py:240-252`): Recursively extracts all string values from nested JSON structures (used for LLM response parsing)
- **`build_markdown_from_inline()`** (`main.py:445-464`): Reconstructs markdown syntax from inline token sequences

## Build & Development Commands

```bash
# Install in editable mode (required for development)
pip install -e .

# Run a specific command
polyglot-rpg init example_project/test_project
polyglot-rpg create-glossary example_project/test_project --use-llm --pre-translate
polyglot-rpg translate example_project/test_project

# Check installation
which polyglot-rpg
polyglot-rpg --help
```

## Configuration

The config file (`01_config.yaml`) contains:

- **api**: URL, API key, model name, temperature for LLM
- **translation_settings**: System prompt for main translation
- **glossary_settings**: Three prompts for extraction, filtering, and pre-translation stages

Template is in `/polyglot_rpg/default_config_template.yaml`. Default setup uses Ollama at `http://localhost:11434/v1` with `gemma3:27b`.

## Key Implementation Details

### AST-Based Markdown Parsing

The tool parses Markdown into AST tokens rather than simple regex. This preserves formatting perfectly:
- Extracts all text content from relevant token types (paragraph, list items, inline text)
- Translates text chunks independently
- Reconstructs markdown by updating token content and rebuilding from AST

### Translation Strategy

1. **Glossary Pre-processing**: Before sending to LLM, glossary terms are substituted in the source text
2. **Caching**: All translation results cached by SHA256 hash of input
3. **Error Handling**: If LLM response contains failure phrases (e.g., "I'm ready when you are"), returns original text
4. **Cost Tracking**: All API calls tracked for token and cost reporting

### User Control Points

- **Glossary review**: Must explicitly rename `2_glossary.for_review.yaml` → `2_glossary.final.yaml` to proceed
- **File selection**: Interactive prompt to choose which chapters to translate
- **Iterative refinement**: Cache prevents re-processing, allowing incremental improvements

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

## Important Notes

- **Python 3.8+** required (uses f-strings, type hints, Path)
- **No tests exist** — add tests if modifying core translation logic
- **Error recovery**: Code is defensive with try-except blocks around LLM calls; original text returned on failure
- **Multiline tokens**: The AST reconstruction handles complex nested markdown (tables, nested lists, etc.)
- **Language support**: Currently hardcoded for English→Russian in prompts, but easily configurable via `01_config.yaml`

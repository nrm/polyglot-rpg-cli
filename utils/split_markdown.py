#!/usr/bin/env python3
"""
Split markdown files into chapters based on major heading structure.
Uses metadata.json (from marker) for validation and context.

This is a standalone wrapper around polyglot_rpg.markdown_utils.MarkdownSplitter.

Usage:
    python3 split_markdown.py <path_to_markdown> <path_to_metadata>

Example:
    python3 split_markdown.py document.md document_meta.json
"""

import sys
from pathlib import Path

# Import from the main package
try:
    from polyglot_rpg.markdown_utils import MarkdownSplitter
except ImportError:
    print("Error: polyglot_rpg package not found. Install it with: pip install -e .")
    sys.exit(1)

def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("📖 Markdown Splitter (post-processor for marker output)")
        print()
        print("Usage:")
        print("  python3 split_markdown.py <markdown_file> <metadata_json>")
        print()
        print("Examples:")
        print("  python3 split_markdown.py document.md document_meta.json")
        print()
        print("Arguments:")
        print("  markdown_file   - Path to markdown file (output from marker)")
        print("  metadata_json   - Path to metadata.json (from marker)")
        print()
        print("Output:")
        print("  Creates 02_input_chapters/ directory with split markdown files")
        print()
        print("Note: You can also use the CLI command: polyglot-rpg split-markdown <markdown_file> <metadata_json>")
        return False

    md_file = Path(sys.argv[1]).resolve()
    meta_file = Path(sys.argv[2]).resolve()
    output_dir = md_file.parent / '02_input_chapters'

    if not md_file.exists():
        print(f"❌ Error: {md_file} not found")
        return False

    if not meta_file.exists():
        print(f"❌ Error: {meta_file} not found")
        return False

    print("=" * 70)
    print("📖 MARKDOWN SPLITTER - Post-processor for marker output")
    print("=" * 70)
    print()

    # Create splitter and split
    splitter = MarkdownSplitter(md_file, meta_file)
    headings = splitter.find_major_headings()

    print(f"🔍 Found {len(headings)} major headings (# level):\n")
    for i, (line_num, title) in enumerate(headings):
        print(f"  {i+1}. Line {line_num+1:4d}: {title}")

    chapters = splitter.split_chapters()

    # Validate
    is_valid, stats = splitter.validate_chapters(chapters)

    print(f"\n✅ VALIDATION\n")
    print(f"  Original file:    {stats['original_chars']:,} characters")
    print(f"  Sum of chapters:  {stats['total_chars']:,} characters")
    print(f"  Difference:       {stats['diff']} characters")

    if not is_valid:
        print(f"\n❌ Validation failed!")
        return False

    if stats['diff'] > 0:
        print(f"  ⚠️  Small difference detected (likely trailing newlines) - acceptable")
    else:
        print(f"  ✅ All content preserved exactly!")

    # Check chapter sizes
    print(f"\n📊 Chapter sizes:\n")
    for name, ch_stats in stats['chapters'].items():
        print(f"  {name:25s}: {ch_stats['chars']:6,} chars, {ch_stats['lines']:4d} lines")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write chapters
    print(f"\n📝 Writing chapters to {output_dir.name}/\n")
    for name, content in chapters.items():
        output_file = output_dir / f"{name}.md"
        output_file.write_text(content, encoding='utf-8')
        print(f"  ✅ {output_file.name}")

    print(f"\n" + "=" * 70)
    print(f"✅ SUCCESS! Created {len(chapters)} chapter files in {output_dir}")
    print(f"=" * 70)
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

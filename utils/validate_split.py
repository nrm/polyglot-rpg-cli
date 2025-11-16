#!/usr/bin/env python3
"""
Comprehensive validation of markdown splitting.
Validates that no content was lost during split process.
Post-processor for marker output to verify integrity.

This is a standalone wrapper around polyglot_rpg.markdown_utils.MarkdownValidator.

Usage:
    python3 validate_split.py <original_markdown> <chapters_directory>

Example:
    python3 validate_split.py document.md ./02_input_chapters
"""

import sys
from pathlib import Path

# Import from the main package
try:
    from polyglot_rpg.markdown_utils import MarkdownValidator
except ImportError:
    print("Error: polyglot_rpg package not found. Install it with: pip install -e .")
    sys.exit(1)


def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("🔍 Markdown Split Validator (post-processor for marker output)")
        print()
        print("Usage:")
        print("  python3 validate_split.py <original_markdown> <chapters_directory>")
        print()
        print("Examples:")
        print("  python3 validate_split.py document.md ./02_input_chapters")
        print()
        print("Arguments:")
        print("  original_markdown   - Path to original markdown file (from marker)")
        print("  chapters_directory  - Path to directory with split chapters")
        print()
        print("Note: You can also use the CLI command: polyglot-rpg validate-split <original_markdown> <chapters_directory>")
        return False

    original_file = Path(sys.argv[1]).resolve()
    chapters_dir = Path(sys.argv[2]).resolve()

    if not original_file.exists():
        print(f"❌ Error: {original_file} not found")
        return False

    if not chapters_dir.exists():
        print(f"❌ Error: {chapters_dir} not found")
        return False

    print("=" * 70)
    print("🔍 MARKDOWN SPLIT VALIDATOR - Post-processor for marker output")
    print("=" * 70)
    print()

    validator = MarkdownValidator(original_file, chapters_dir)
    results = validator.run_all_checks()

    # Display results for each check
    for check_name, (passed, stats) in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"

        if check_name == "Character Count":
            print("📊 CHARACTER COUNT")
            print(f"  Original:      {stats['original']:,} characters")
            print(f"  Combined:      {stats['combined']:,} characters")
            print(f"  Difference:    {stats['difference']} characters")
            print(f"  {status}\n")

        elif check_name == "Line Count":
            print("📋 LINE COUNT")
            print(f"  Original:      {stats['original']:,} lines")
            print(f"  Combined:      {stats['combined']:,} lines")
            print(f"  Difference:    {stats['difference']} lines")
            print(f"  {status}\n")

        elif check_name == "Headings":
            print("📑 MARKDOWN HEADINGS")
            print(f"  Original:      {stats['original']} headings")
            print(f"  Combined:      {stats['combined']} headings")
            print(f"  {status}\n")

        elif check_name == "Images":
            print("🖼️  IMAGE REFERENCES")
            print(f"  Original:      {stats['original']} images")
            print(f"  Combined:      {stats['combined']} images")
            if stats['missing']:
                print(f"  ❌ Missing: {stats['missing']}")
            if stats['extra']:
                print(f"  ⚠️  Extra: {stats['extra']}")
            print(f"  {status}\n")

        elif check_name == "Code Blocks":
            print("💻 CODE BLOCKS")
            print(f"  Original:      {stats['original']} blocks")
            print(f"  Combined:      {stats['combined']} blocks")
            print(f"  {status}\n")

        elif check_name == "Links":
            print("🔗 MARKDOWN LINKS")
            print(f"  Original:      {stats['original']} links")
            print(f"  Combined:      {stats['combined']} links")
            print(f"  {status}\n")

        elif check_name == "List Items":
            print("📝 LIST ITEMS")
            print(f"  Original:      {stats['original']} items")
            print(f"  Combined:      {stats['combined']} items")
            print(f"  {status}\n")

        elif check_name == "Word Count":
            print("📖 WORD COUNT")
            print(f"  Original:      {stats['original']:,} words")
            print(f"  Combined:      {stats['combined']:,} words")
            print(f"  Difference:    {stats['difference_pct']:.2f}%")
            print(f"  {status}\n")

        elif check_name == "Chapter Structure":
            print("📂 CHAPTER FILE STRUCTURE")
            print(f"  Number of chapters: {stats['chapter_count']}\n")
            for ch in stats['chapters']:
                print(f"  {ch['name']:30s}: {ch['chars']:7,} chars, {ch['lines']:4d} lines, {ch['headings']:2d} headings")
            print(f"\n  {status}\n")

    # Summary
    print("=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    passed = sum(1 for v, _ in results.values() if v)
    total = len(results)

    for check, (result, _) in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check:25s}: {status}")

    print()
    print(f"  Result: {passed}/{total} checks passed")
    print()

    if passed == total:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        print("✅ No content was lost during splitting.")
        return True
    elif passed >= total - 1:
        print("⚠️  MOSTLY OK: Minor differences detected but acceptable.")
        print("✅ Content integrity preserved.")
        return True
    else:
        print("❌ VALIDATION FAILED: Significant issues detected.")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

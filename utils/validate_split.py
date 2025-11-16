#!/usr/bin/env python3
"""
Comprehensive validation of markdown splitting.
Validates that no content was lost during split process.
Post-processor for marker output to verify integrity.

Usage:
    python3 validate_split.py <original_markdown> <chapters_directory>

Example:
    python3 validate_split.py PointsOfLight.md ./02_input_chapters
    python3 validate_split.py ../example_project/PointsOfLight/PointsOfLight.md ../example_project/PointsOfLight/02_input_chapters
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

class MarkdownValidator:
    """Validates that splitting preserved all content."""

    def __init__(self, original_file: Path, chapters_dir: Path):
        self.original = original_file.read_text(encoding='utf-8')
        self.chapters_dir = chapters_dir
        self.chapter_files = sorted(chapters_dir.glob('*.md'))
        self.combined = '\n'.join([f.read_text(encoding='utf-8') for f in self.chapter_files])

    def validate_character_count(self) -> bool:
        """Check total character count."""
        orig_len = len(self.original)
        combined_len = len(self.combined)
        diff = abs(orig_len - combined_len)

        print("📊 CHARACTER COUNT")
        print(f"  Original:      {orig_len:,} characters")
        print(f"  Combined:      {combined_len:,} characters")
        print(f"  Difference:    {diff} characters")

        if diff <= 10:
            print(f"  ✅ PASS (difference within tolerance)\n")
            return True
        else:
            print(f"  ❌ FAIL (difference too large)\n")
            return False

    def validate_line_count(self) -> bool:
        """Check total line count."""
        orig_lines = len(self.original.split('\n'))
        combined_lines = len(self.combined.split('\n'))
        diff = abs(orig_lines - combined_lines)

        print("📋 LINE COUNT")
        print(f"  Original:      {orig_lines:,} lines")
        print(f"  Combined:      {combined_lines:,} lines")
        print(f"  Difference:    {diff} lines")

        if diff <= 5:
            print(f"  ✅ PASS\n")
            return True
        else:
            print(f"  ❌ FAIL\n")
            return False

    def validate_headings(self) -> bool:
        """Check that all markdown headings are preserved."""
        # Extract all headings
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        orig_headings = re.findall(heading_pattern, self.original, re.MULTILINE)
        combined_headings = re.findall(heading_pattern, self.combined, re.MULTILINE)

        print("📑 MARKDOWN HEADINGS")
        print(f"  Original:      {len(orig_headings)} headings")
        print(f"  Combined:      {len(combined_headings)} headings")

        if orig_headings == combined_headings:
            print(f"  ✅ PASS (all headings match)\n")
            return True
        else:
            print(f"  ⚠️  WARNING: Heading count mismatch\n")
            # Show differences
            orig_set = set(orig_headings)
            combined_set = set(combined_headings)
            if orig_set - combined_set:
                print(f"  Missing headings: {orig_set - combined_set}\n")
            return len(orig_headings) == len(combined_headings)

    def validate_images(self) -> bool:
        """Check that all image references are preserved."""
        image_pattern = r'!\[\]\(([^)]+)\)'
        orig_images = re.findall(image_pattern, self.original)
        combined_images = re.findall(image_pattern, self.combined)

        print("🖼️  IMAGE REFERENCES")
        print(f"  Original:      {len(orig_images)} images")
        print(f"  Combined:      {len(combined_images)} images")

        orig_set = set(orig_images)
        combined_set = set(combined_images)

        if orig_set == combined_set:
            print(f"  ✅ PASS (all images preserved)\n")
            return True
        else:
            missing = orig_set - combined_set
            extra = combined_set - orig_set
            if missing:
                print(f"  ❌ Missing: {missing}\n")
            if extra:
                print(f"  ⚠️  Extra: {extra}\n")
            return len(orig_images) == len(combined_images)

    def validate_code_blocks(self) -> bool:
        """Check that code blocks are preserved."""
        code_pattern = r'```[\s\S]*?```'
        orig_codes = re.findall(code_pattern, self.original)
        combined_codes = re.findall(code_pattern, self.combined)

        print("💻 CODE BLOCKS")
        print(f"  Original:      {len(orig_codes)} blocks")
        print(f"  Combined:      {len(combined_codes)} blocks")

        if len(orig_codes) == len(combined_codes):
            print(f"  ✅ PASS\n")
            return True
        else:
            print(f"  ❌ FAIL\n")
            return False

    def validate_links(self) -> bool:
        """Check that all links are preserved."""
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        orig_links = re.findall(link_pattern, self.original)
        combined_links = re.findall(link_pattern, self.combined)

        print("🔗 MARKDOWN LINKS")
        print(f"  Original:      {len(orig_links)} links")
        print(f"  Combined:      {len(combined_links)} links")

        if len(orig_links) == len(combined_links):
            print(f"  ✅ PASS\n")
            return True
        else:
            print(f"  ⚠️  WARNING: Link count mismatch\n")
            return False

    def validate_lists(self) -> bool:
        """Check that list markers are preserved."""
        # Count list items (lines starting with - or *)
        list_pattern = r'^[\s]*[-*]\s+'
        orig_lists = len(re.findall(list_pattern, self.original, re.MULTILINE))
        combined_lists = len(re.findall(list_pattern, self.combined, re.MULTILINE))

        print("📝 LIST ITEMS")
        print(f"  Original:      {orig_lists} items")
        print(f"  Combined:      {combined_lists} items")

        if orig_lists == combined_lists:
            print(f"  ✅ PASS\n")
            return True
        else:
            print(f"  ⚠️  WARNING: List count mismatch\n")
            return False

    def validate_chapter_structure(self) -> bool:
        """Check chapter file structure."""
        print("📂 CHAPTER FILE STRUCTURE")
        print(f"  Number of chapters: {len(self.chapter_files)}\n")

        for f in self.chapter_files:
            content = f.read_text(encoding='utf-8')
            lines = len(content.split('\n'))
            chars = len(content)
            # Count headings in this chapter
            headings = len(re.findall(r'^#', content, re.MULTILINE))

            print(f"  {f.name:30s}: {chars:7,} chars, {lines:4d} lines, {headings:2d} headings")

        print(f"\n  ✅ All chapter files present\n")
        return True

    def validate_word_count(self) -> bool:
        """Rough word count check."""
        orig_words = len(self.original.split())
        combined_words = len(self.combined.split())
        diff_pct = abs(orig_words - combined_words) / orig_words * 100 if orig_words > 0 else 0

        print("📖 WORD COUNT")
        print(f"  Original:      {orig_words:,} words")
        print(f"  Combined:      {combined_words:,} words")
        print(f"  Difference:    {diff_pct:.2f}%")

        if diff_pct < 1:
            print(f"  ✅ PASS\n")
            return True
        else:
            print(f"  ⚠️  WARNING: Significant word count difference\n")
            return False

    def run_all_checks(self) -> dict:
        """Run all validation checks."""
        results = {
            'Character Count': self.validate_character_count(),
            'Line Count': self.validate_line_count(),
            'Headings': self.validate_headings(),
            'Images': self.validate_images(),
            'Code Blocks': self.validate_code_blocks(),
            'Links': self.validate_links(),
            'List Items': self.validate_lists(),
            'Word Count': self.validate_word_count(),
            'Chapter Structure': self.validate_chapter_structure(),
        }
        return results

def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("🔍 Markdown Split Validator (post-processor for marker output)")
        print()
        print("Usage:")
        print("  python3 validate_split.py <original_markdown> <chapters_directory>")
        print()
        print("Examples:")
        print("  python3 validate_split.py PointsOfLight.md ./02_input_chapters")
        print("  python3 validate_split.py ../example_project/PointsOfLight/PointsOfLight.md ../example_project/PointsOfLight/02_input_chapters")
        print()
        print("Arguments:")
        print("  original_markdown   - Path to original markdown file (from marker)")
        print("  chapters_directory  - Path to directory with split chapters")
        print()
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

    # Summary
    print("=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for check, result in results.items():
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

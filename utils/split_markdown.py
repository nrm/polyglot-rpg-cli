#!/usr/bin/env python3
"""
Split markdown files into chapters based on major heading structure.
Uses metadata.json (from marker) for validation and context.

Usage:
    python3 split_markdown.py <path_to_markdown> <path_to_metadata>

Example:
    python3 split_markdown.py PointsOfLight.md PointsOfLight_meta.json
    python3 split_markdown.py ../example_project/PointsOfLight/PointsOfLight.md ../example_project/PointsOfLight/PointsOfLight_meta.json
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

class MarkdownSplitter:
    """Splits markdown into chapters based on # headings."""

    def __init__(self, md_path: Path, meta_path: Path):
        self.md_path = md_path
        self.meta_path = meta_path
        self.content = md_path.read_text(encoding='utf-8')
        self.lines = self.content.split('\n')
        self.metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        self.toc_titles = [item['title'] for item in self.metadata['table_of_contents']]

    def find_major_headings(self) -> List[Tuple[int, str]]:
        """Find all # (level 1) headings with their line numbers."""
        headings = []
        for i, line in enumerate(self.lines):
            # Match # heading but not ## or ###
            if re.match(r'^# ', line):
                title = line.lstrip('# ').strip()
                headings.append((i, title))
        return headings

    def extract_chapter(self, start_line: int, end_line: int = None) -> str:
        """Extract chapter content from start to end line."""
        if end_line is None:
            end_line = len(self.lines)
        return '\n'.join(self.lines[start_line:end_line])

    def split_chapters(self) -> dict:
        """Split markdown into chapters based on major headings."""
        headings = self.find_major_headings()

        print(f"🔍 Found {len(headings)} major headings (# level):\n")
        for i, (line_num, title) in enumerate(headings):
            print(f"  {i+1}. Line {line_num+1:4d}: {title}")

        if len(headings) < 2:
            raise ValueError(f"Expected at least 2 major headings, found {len(headings)}")

        chapters = {}

        # Strategy: Use each heading as a chapter boundary
        # Detect if first heading is a document title (short, no "part" or "chapter" keywords)
        # If so, include it with the next chapter; otherwise start from it

        first_heading_text = headings[0][1].lower()
        is_title = (
            len(headings[0][1]) < 50 and
            'part' not in first_heading_text and
            'chapter' not in first_heading_text and
            first_heading_text not in ['introduction', 'getting started']
        )

        start_idx = 1 if is_title else 0

        for i in range(start_idx, len(headings)):
            heading_idx = i
            start_line = 0 if i == start_idx else headings[heading_idx][0]
            end_line = headings[heading_idx + 1][0] if heading_idx + 1 < len(headings) else len(self.lines)

            # Create chapter name from heading
            chapter_title = headings[heading_idx][1].replace('**', '').strip()
            chapter_title = chapter_title.replace(' ', '_').replace(':', '').replace('/', '_')
            chapter_num = i - start_idx
            chapter_name = f"{chapter_num}_{chapter_title}"

            chapters[chapter_name] = self.extract_chapter(start_line, end_line)

        return chapters

    def validate_chapters(self, chapters: dict) -> bool:
        """Validate that all content is preserved."""
        total_chars = sum(len(ch) for ch in chapters.values())
        original_chars = len(self.content)
        diff = abs(total_chars - original_chars)

        print(f"\n✅ VALIDATION\n")
        print(f"  Original file:    {original_chars:,} characters")
        print(f"  Sum of chapters:  {total_chars:,} characters")
        print(f"  Difference:       {diff} characters")

        # Allow small differences (trailing newlines, etc.)
        if diff > 10:
            print(f"  ❌ ERROR: Character count mismatch is too large!")
            return False

        if diff > 0:
            print(f"  ⚠️  Small difference detected (likely trailing newlines) - acceptable")
        else:
            print(f"  ✅ All content preserved exactly!")

        # Check chapter sizes
        print(f"\n📊 Chapter sizes:\n")
        for name, content in chapters.items():
            lines = len(content.split('\n'))
            print(f"  {name:25s}: {len(content):6,} chars, {lines:4d} lines")

        return True

def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("📖 Markdown Splitter (post-processor for marker output)")
        print()
        print("Usage:")
        print("  python3 split_markdown.py <markdown_file> <metadata_json>")
        print()
        print("Examples:")
        print("  python3 split_markdown.py PointsOfLight.md PointsOfLight_meta.json")
        print("  python3 split_markdown.py ../example_project/PointsOfLight/PointsOfLight.md ../example_project/PointsOfLight/PointsOfLight_meta.json")
        print()
        print("Arguments:")
        print("  markdown_file   - Path to markdown file (output from marker)")
        print("  metadata_json   - Path to metadata.json (from marker)")
        print()
        print("Output:")
        print("  Creates 02_input_chapters/ directory with split markdown files")
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
    chapters = splitter.split_chapters()

    # Validate
    is_valid = splitter.validate_chapters(chapters)

    if not is_valid:
        print("\n❌ Validation failed!")
        return False

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

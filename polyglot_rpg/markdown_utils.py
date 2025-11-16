#!/usr/bin/env python3
"""
Markdown utilities for splitting and validating markdown files.
Post-processing tools for marker (PDF-to-Markdown) output.
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class MarkdownSplitter:
    """Splits markdown into chapters based on # headings."""

    def __init__(self, md_path: Path, meta_path: Optional[Path] = None):
        self.md_path = md_path
        self.meta_path = meta_path
        self.content = md_path.read_text(encoding='utf-8')
        self.lines = self.content.split('\n')

        # Metadata is optional - only needed if you want access to marker's table_of_contents
        self.metadata = None
        self.toc_titles = []
        if meta_path and meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding='utf-8'))
            self.toc_titles = [item['title'] for item in self.metadata.get('table_of_contents', [])]

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

    def split_chapters(self) -> Dict[str, str]:
        """Split markdown into chapters based on major headings."""
        headings = self.find_major_headings()

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

    def validate_chapters(self, chapters: Dict[str, str]) -> Tuple[bool, Dict[str, any]]:
        """
        Validate that all content is preserved.
        Returns (is_valid, stats_dict)
        """
        total_chars = sum(len(ch) for ch in chapters.values())
        original_chars = len(self.content)
        diff = abs(total_chars - original_chars)

        stats = {
            'original_chars': original_chars,
            'total_chars': total_chars,
            'diff': diff,
            'chapters': {
                name: {
                    'chars': len(content),
                    'lines': len(content.split('\n'))
                }
                for name, content in chapters.items()
            }
        }

        # Allow small differences (trailing newlines, etc.)
        is_valid = diff <= 10

        return is_valid, stats


class MarkdownValidator:
    """Validates that splitting preserved all content."""

    def __init__(self, original_file: Path, chapters_dir: Path):
        self.original = original_file.read_text(encoding='utf-8')
        self.chapters_dir = chapters_dir
        self.chapter_files = sorted(chapters_dir.glob('*.md'))
        self.combined = '\n'.join([f.read_text(encoding='utf-8') for f in self.chapter_files])

    def validate_character_count(self) -> Tuple[bool, Dict[str, int]]:
        """Check total character count."""
        orig_len = len(self.original)
        combined_len = len(self.combined)
        diff = abs(orig_len - combined_len)

        stats = {
            'original': orig_len,
            'combined': combined_len,
            'difference': diff
        }

        return diff <= 10, stats

    def validate_line_count(self) -> Tuple[bool, Dict[str, int]]:
        """Check total line count."""
        orig_lines = len(self.original.split('\n'))
        combined_lines = len(self.combined.split('\n'))
        diff = abs(orig_lines - combined_lines)

        stats = {
            'original': orig_lines,
            'combined': combined_lines,
            'difference': diff
        }

        return diff <= 5, stats

    def validate_headings(self) -> Tuple[bool, Dict]:
        """Check that all markdown headings are preserved."""
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        orig_headings = re.findall(heading_pattern, self.original, re.MULTILINE)
        combined_headings = re.findall(heading_pattern, self.combined, re.MULTILINE)

        stats = {
            'original': len(orig_headings),
            'combined': len(combined_headings),
            'match': orig_headings == combined_headings
        }

        return len(orig_headings) == len(combined_headings), stats

    def validate_images(self) -> Tuple[bool, Dict]:
        """Check that all image references are preserved."""
        image_pattern = r'!\[\]\(([^)]+)\)'
        orig_images = re.findall(image_pattern, self.original)
        combined_images = re.findall(image_pattern, self.combined)

        orig_set = set(orig_images)
        combined_set = set(combined_images)

        stats = {
            'original': len(orig_images),
            'combined': len(combined_images),
            'missing': list(orig_set - combined_set),
            'extra': list(combined_set - orig_set)
        }

        return len(orig_images) == len(combined_images), stats

    def validate_code_blocks(self) -> Tuple[bool, Dict]:
        """Check that code blocks are preserved."""
        code_pattern = r'```[\s\S]*?```'
        orig_codes = re.findall(code_pattern, self.original)
        combined_codes = re.findall(code_pattern, self.combined)

        stats = {
            'original': len(orig_codes),
            'combined': len(combined_codes)
        }

        return len(orig_codes) == len(combined_codes), stats

    def validate_links(self) -> Tuple[bool, Dict]:
        """Check that all links are preserved."""
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        orig_links = re.findall(link_pattern, self.original)
        combined_links = re.findall(link_pattern, self.combined)

        stats = {
            'original': len(orig_links),
            'combined': len(combined_links)
        }

        return len(orig_links) == len(combined_links), stats

    def validate_lists(self) -> Tuple[bool, Dict]:
        """Check that list markers are preserved."""
        list_pattern = r'^[\s]*[-*]\s+'
        orig_lists = len(re.findall(list_pattern, self.original, re.MULTILINE))
        combined_lists = len(re.findall(list_pattern, self.combined, re.MULTILINE))

        stats = {
            'original': orig_lists,
            'combined': combined_lists
        }

        return orig_lists == combined_lists, stats

    def validate_word_count(self) -> Tuple[bool, Dict]:
        """Rough word count check."""
        orig_words = len(self.original.split())
        combined_words = len(self.combined.split())
        diff_pct = abs(orig_words - combined_words) / orig_words * 100 if orig_words > 0 else 0

        stats = {
            'original': orig_words,
            'combined': combined_words,
            'difference_pct': diff_pct
        }

        return diff_pct < 1, stats

    def validate_chapter_structure(self) -> Tuple[bool, Dict]:
        """Check chapter file structure."""
        chapters_info = []

        for f in self.chapter_files:
            content = f.read_text(encoding='utf-8')
            lines = len(content.split('\n'))
            chars = len(content)
            headings = len(re.findall(r'^#', content, re.MULTILINE))

            chapters_info.append({
                'name': f.name,
                'chars': chars,
                'lines': lines,
                'headings': headings
            })

        stats = {
            'chapter_count': len(self.chapter_files),
            'chapters': chapters_info
        }

        return True, stats

    def run_all_checks(self) -> Dict[str, Tuple[bool, Dict]]:
        """Run all validation checks. Returns dict of check_name -> (passed, stats)"""
        return {
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

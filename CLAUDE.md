# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ebook image processing and conversion toolkit combining:
1. Python-based image processing pipeline for manga/book pages
2. Browser-based ZIP-to-EPUB converter web application

The project is bilingual (English/Japanese) focused on Japanese ebook formatting (right-to-left binding, vertical text).

## Core Components

### Python Image Processing Scripts

**improved_image_cropper.py** - Main interactive image cropping tool
- Removes margins from scanned book pages with configurable settings
- Interactive CLI prompts for: crop margins (%), page number removal (top/bottom/both), aspect ratio unification
- Processing pipeline: custom margin crop → white border removal (threshold=200) → optional page number removal → add 5px margin → optional aspect ratio padding
- Creates detailed logs in parent folder: `image_cropper_log.txt`
- Output: `{input_folder}/cropped/`

**AddPadding_average.py** - Aspect ratio unification with advanced padding
- Calculates target aspect ratio from odd/even page folders (excludes outlier quartiles)
- Three margin patterns for different binding styles:
  - Pattern 1: Top-aligned padding (horizontal writing)
  - Pattern 2: Odd pages right-aligned, even pages left-aligned (vertical writing, spread layout)
  - Pattern 3: Odd pages left-aligned, even pages right-aligned (vertical writing, alternate spread)
- Expected folder structure: `{parent}/odd/cropped/`, `{parent}/even/cropped/`, `{parent}/odd/ex/resized/`, `{parent}/even/ex/resized/`
- Output: `{parent}/margin_added/`

**zip_to_epub.py** - Command-line EPUB generator
- Converts image ZIP files to EPUB format with Japanese settings (RTL)
- Auto-generates cover image from title using Japanese fonts (MS Gothic/Meiryo/Mincho)
- Hardcoded defaults: author="hogehoge", publisher="hogehoge"
- Dependencies: PIL, ebooklib, zipfile

**comp_v5.py** - Image compression utility (latest version)

**config.json** - Configuration for image processing
- `threshold`: 120 (white detection)
- `min_margin`: 0
- `debug_mode`: true

### Web Application (Browser-based)

**index.html + app.js** - ZIP to EPUB converter
- Drag-and-drop ZIP file upload
- Client-side processing using JSZip library (v3.10.1)
- Binding direction selection (RTL for manga, LTR for horizontal text)
- Pure client-side implementation (no server upload)
- Deployable to GitHub Pages

## Development Commands

### Running Image Processing

```bash
# Interactive image cropper (main tool)
python improved_image_cropper.py

# Add padding with aspect ratio unification
python AddPadding_average.py

# Create EPUB from ZIP
python zip_to_epub.py

# Compress images
python comp_v5.py
```

### Python Dependencies

```bash
pip install pillow numpy ebooklib
```

### Web Application

Open `index.html` in browser - no build step required.

For GitHub Pages deployment:
1. Push to GitHub repository
2. Enable Pages in repository settings (Settings → Pages)
3. Select main/master branch as source

## Important Notes

- All scripts expect input via CLI prompts with drag-and-drop path support (auto-strips quotes)
- Image processing uses noise-skipping algorithm (5px threshold) to handle scanning artifacts
- Logs are automatically generated with timestamps in parent folders
- Scripts preserve JPEG quality (quality=95) during processing
- Default margin handling: 5% crop, 200 luminance threshold for white detection
- Odd/even page separation is manual (pre-process images into separate folders)

## File Naming Conventions

- Versioned scripts: `comp_v3.py`, `comp_v4.py`, `comp_v5.py` (use latest v5)
- Image files are sorted numerically for EPUB page order
- Log files: `image_cropper_log.txt`, `log.txt` (in parent directories)

## Architecture Patterns

The codebase follows a functional, script-based architecture:
- Each Python script is self-contained with `if __name__ == "__main__"` entry points
- Interactive CLI workflows with input validation and defaults
- Modular functions for image operations (crop, pad, detect borders)
- Logging pattern: initialize_log() → write_log() throughout processing → completion timestamp
- Error handling with counters for batch processing statistics

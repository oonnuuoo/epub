#!/usr/bin/env python3
"""
PDF Embedded Image Extractor
Extracts all raster images embedded inside a PDF file.

Usage:
  python pdf_extract_embedded_images.py input.pdf
  python pdf_extract_embedded_images.py input.pdf -o output_dir
  python pdf_extract_embedded_images.py input.pdf --pages 1-5
  python pdf_extract_embedded_images.py input.pdf --min-size 5000
  python pdf_extract_embedded_images.py input.pdf --format png
"""

import argparse
import sys
from pathlib import Path


# ── Dependency check ──────────────────────────────────────────────────────────

def ensure_pymupdf():
    try:
        import fitz  # noqa: F401
    except ImportError:
        print("PyMuPDF not found. Installing...")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pymupdf", "--quiet"],
            check=True,
        )
        print("PyMuPDF installed.\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_pages(spec: str, total: int) -> set[int]:
    """Parse a page spec like '1-3,5,7-9' into a set of 0-based indices."""
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a) - 1, int(b)))
        else:
            pages.add(int(part) - 1)
    return {p for p in pages if 0 <= p < total}


def colorspace_name(cs) -> str:
    import fitz
    mapping = {
        fitz.csGRAY: "GRAY",
        fitz.csRGB:  "RGB",
        fitz.csCMYK: "CMYK",
    }
    return mapping.get(cs, "OTHER")


# ── Core extractor ────────────────────────────────────────────────────────────

def extract_images(
    pdf_path: Path,
    out_dir: Path,
    page_spec: str | None,
    min_size: int,
    fmt: str,
) -> None:
    import fitz

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    target_pages = (
        parse_pages(page_spec, total_pages)
        if page_spec
        else set(range(total_pages))
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    ext = fmt.lower()
    saved = 0
    skipped = 0

    print(f"PDF     : {pdf_path.name}")
    print(f"Pages   : {total_pages}  (scanning {len(target_pages)})")
    print(f"Output  : {out_dir}")
    print(f"Min size: {min_size:,} bytes")
    print(f"Format  : {ext.upper()}\n")

    seen_xrefs: set[int] = set()  # avoid saving the same image twice

    for page_num in sorted(target_pages):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        if not image_list:
            continue

        print(f"  Page {page_num + 1:>4}  —  {len(image_list)} image object(s) found")

        for img_info in image_list:
            xref = img_info[0]

            if xref in seen_xrefs:
                print(f"           xref={xref}  skip (duplicate)")
                continue
            seen_xrefs.add(xref)

            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception as e:
                print(f"           xref={xref}  skip (could not decode: {e})")
                skipped += 1
                continue

            # Convert exotic colourspaces (CMYK, etc.) to RGB
            if pix.colorspace and pix.colorspace not in (fitz.csGRAY, fitz.csRGB):
                pix = fitz.Pixmap(fitz.csRGB, pix)

            # Skip tiny images (masks, icons, backgrounds)
            if pix.size < min_size:
                print(
                    f"           xref={xref}  skip "
                    f"({pix.width}×{pix.height}, {pix.size:,} bytes < min)"
                )
                skipped += 1
                pix = None
                continue

            cs = colorspace_name(pix.colorspace)
            fname = f"page{page_num + 1:04d}_xref{xref}.{ext}"
            out_path = out_dir / fname

            try:
                if ext in ("jpg", "jpeg"):
                    pix.save(str(out_path), jpg_quality=92)
                else:
                    pix.save(str(out_path))
                print(
                    f"           xref={xref}  saved  "
                    f"{pix.width}×{pix.height} {cs}  {out_path.stat().st_size:,} B  → {fname}"
                )
                saved += 1
            except Exception as e:
                print(f"           xref={xref}  error saving: {e}")
                skipped += 1
            finally:
                pix = None  # free memory

    doc.close()

    print(f"\n{'─'*60}")
    print(f"Done.  Saved: {saved}  Skipped: {skipped}")
    print(f"Output directory: {out_dir.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract all raster images embedded in a PDF file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_extract_embedded_images.py report.pdf
  python pdf_extract_embedded_images.py report.pdf -o ./images
  python pdf_extract_embedded_images.py report.pdf --pages 1-10
  python pdf_extract_embedded_images.py report.pdf --min-size 10000
  python pdf_extract_embedded_images.py report.pdf --format jpg
""",
    )
    p.add_argument("pdf", help="Path to the input PDF file")
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: <pdf_name>_images/)",
    )
    p.add_argument(
        "--pages",
        default=None,
        metavar="SPEC",
        help="Pages to scan, e.g. '1-5' or '1,3,7-10' (default: all)",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=2000,
        metavar="BYTES",
        help="Skip images smaller than this byte size (default: 2000). "
             "Useful to filter out masks and decorative elements.",
    )
    p.add_argument(
        "--format",
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="Output image format (default: png)",
    )
    return p


def main():
    ensure_pymupdf()

    parser = build_parser()
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"Error: file not found — {pdf_path}")
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print(f"Warning: file does not have a .pdf extension — {pdf_path.name}")

    out_dir = Path(args.output) if args.output else pdf_path.parent / (pdf_path.stem + "_images")

    extract_images(
        pdf_path=pdf_path,
        out_dir=out_dir,
        page_spec=args.pages,
        min_size=args.min_size,
        fmt=args.format,
    )


if __name__ == "__main__":
    main()
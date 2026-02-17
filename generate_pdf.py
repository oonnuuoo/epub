import glob
import os
import sys
import csv
from fpdf import FPDF
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NumberObject,
)
from PIL import Image

# Usage: python generate_pdf.py <folder_path>
# folder_path: absolute or relative path to the book directory
#              (must contain comp/ with JPG images and displacement.csv)
if len(sys.argv) < 2:
    print("Usage: python generate_pdf.py <folder_path>")
    print("  folder_path: path to the book directory containing comp/ and displacement.csv")
    sys.exit(1)

BASE_DIR = os.path.abspath(sys.argv[1])
input_csv = os.path.join(BASE_DIR, 'displacement.csv')
COMP_DIR = os.path.join(BASE_DIR, 'comp')
OUTPUT_PDF = os.path.join(BASE_DIR, 'output.pdf')
TEMP_PDF = os.path.join(BASE_DIR, '_temp.pdf')

# 1. Collect and sort images by name ascending
images = sorted(glob.glob(os.path.join(COMP_DIR, "*.jpg")))
print(f"Found {len(images)} images")
for img_path in images[:10]:
    print(f"  {os.path.basename(img_path)}")
print("  ...")

# Build a mapping from image filename to page index (0-based)
img_to_page = {}
for i, img_path in enumerate(images):
    img_to_page[os.path.basename(img_path)] = i

# 2. Build PDF from images using fpdf2
pdf = FPDF(unit="pt")
pdf.set_auto_page_break(False)

for img_path in images:
    with Image.open(img_path) as img:
        w_px, h_px = img.size
    page_w = 595.0
    scale = page_w / w_px
    page_h = h_px * scale

    pdf.add_page(format=(page_w, page_h))
    pdf.image(img_path, x=0, y=0, w=page_w, h=page_h)

pdf.output(TEMP_PDF)
print(f"Temporary PDF created: {TEMP_PDF}")

# 3. Read PDF and add link annotations from CSV
reader = PdfReader(TEMP_PDF)
writer = PdfWriter()

for page in reader.pages:
    writer.add_page(page)

# Read CSV and add link annotations
# CSV columns: source_img, x, y, w, h(tentative), h, target_page_num, target_img
with open(input_csv, 'r', encoding='utf-8-sig') as f:
    csv_reader = csv.reader(f)
    header = next(csv_reader)  # skip header row

    for row in csv_reader:
        if not row or not row[0]:
            continue

        source_img_name = row[0]
        x = int(row[1])
        y = int(row[2])
        w = int(row[3])
        h = int(row[5])  # use actual height (column 5)
        target_img_name = row[7]

        # Find source page index
        if source_img_name not in img_to_page:
            print(f"  Warning: source image {source_img_name} not found, skipping")
            continue
        source_page_idx = img_to_page[source_img_name]

        # Find target page index
        if target_img_name not in img_to_page:
            print(f"  Warning: target image {target_img_name} not found, skipping")
            continue
        target_page_idx = img_to_page[target_img_name]

        # Get source page and compute scale
        source_page = writer.pages[source_page_idx]
        pdf_page_w = float(source_page.mediabox.width)
        pdf_page_h = float(source_page.mediabox.height)

        # Get original image dimensions for scale calculation
        img_path = os.path.join(COMP_DIR, source_img_name)
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        scale_x = pdf_page_w / img_w
        scale_y = pdf_page_h / img_h

        # Convert image pixel coords to PDF coords
        # PDF: origin at bottom-left, y increases upward
        x1 = x * scale_x
        x2 = (x + w) * scale_x
        y1_pdf = pdf_page_h - ((y + h) * scale_y)  # bottom in PDF coords
        y2_pdf = pdf_page_h - (y * scale_y)         # top in PDF coords

        # Create link annotation
        target_page = writer.pages[target_page_idx]
        link_annotation = DictionaryObject()
        link_annotation.update({
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Link"),
            NameObject("/Rect"): ArrayObject([
                FloatObject(x1),
                FloatObject(y1_pdf),
                FloatObject(x2),
                FloatObject(y2_pdf),
            ]),
            NameObject("/Border"): ArrayObject([
                NumberObject(0), NumberObject(0), NumberObject(0)
            ]),
            NameObject("/Dest"): ArrayObject([
                target_page.indirect_reference,
                NameObject("/Fit"),
            ]),
        })

        # Add annotation to source page
        if "/Annots" in source_page:
            source_page[NameObject("/Annots")].append(link_annotation)
        else:
            source_page[NameObject("/Annots")] = ArrayObject([link_annotation])

        print(f"  Link: {source_img_name} (p.{source_page_idx+1}) rect=({x},{y})-({x+w},{y+h}) -> {target_img_name} (p.{target_page_idx+1})")

writer.write(OUTPUT_PDF)
print(f"Output PDF created: {OUTPUT_PDF}")

# Cleanup temp file
os.remove(TEMP_PDF)
print("Done!")

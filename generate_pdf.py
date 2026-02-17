import glob
import os
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

COMP_DIR = os.path.join(os.path.dirname(__file__), "comp")
OUTPUT_PDF = os.path.join(os.path.dirname(__file__), "output.pdf")
TEMP_PDF = os.path.join(os.path.dirname(__file__), "_temp.pdf")

# 1. Collect and sort images by name ascending
images = sorted(glob.glob(os.path.join(COMP_DIR, "*.jpg")))
print(f"Found {len(images)} images")
for img_path in images[:10]:
    print(f"  {os.path.basename(img_path)}")
print("  ...")

# 2. Build PDF from images using fpdf2
pdf = FPDF(unit="pt")
pdf.set_auto_page_break(False)

for img_path in images:
    with Image.open(img_path) as img:
        w_px, h_px = img.size
    # Convert pixels to points at 72 DPI equivalent (fit page)
    # Use a standard approach: set page size to image aspect ratio
    # Scale to reasonable page size (A4-ish width 595pt)
    page_w = 595.0
    scale = page_w / w_px
    page_h = h_px * scale

    pdf.add_page(format=(page_w, page_h))
    pdf.image(img_path, x=0, y=0, w=page_w, h=page_h)

pdf.output(TEMP_PDF)
print(f"Temporary PDF created: {TEMP_PDF}")

# 3. Add link annotation on page 7 (0-indexed: 6) linking to page 12 (0-indexed: 11)
#    Rectangle in original image coords: (79,421)-(1175,2721)
#    Need to convert to PDF points using the same scale factor
reader = PdfReader(TEMP_PDF)
writer = PdfWriter()

for page in reader.pages:
    writer.add_page(page)

# Get page 7 dimensions to compute scale
page7 = writer.pages[6]  # 0-indexed
pdf_page_w = float(page7.mediabox.width)
pdf_page_h = float(page7.mediabox.height)

# The original image for page 7 is 006.jpg
with Image.open(os.path.join(COMP_DIR, "006.jpg")) as img:
    img_w, img_h = img.size

scale_x = pdf_page_w / img_w
scale_y = pdf_page_h / img_h

# Convert image pixel coords to PDF points
# PDF coordinate system: origin at bottom-left, y increases upward
x1 = 79 * scale_x
x2 = 1175 * scale_x
# For y: PDF y = page_height - (image_y * scale_y)
y1_pdf = pdf_page_h - (2721 * scale_y)  # bottom of rect in PDF coords
y2_pdf = pdf_page_h - (421 * scale_y)   # top of rect in PDF coords

print(f"Image size: {img_w}x{img_h}")
print(f"PDF page size: {pdf_page_w}x{pdf_page_h}")
print(f"Link rect (PDF coords): ({x1:.1f}, {y1_pdf:.1f}) - ({x2:.1f}, {y2_pdf:.1f})")

# Target: page 12 (0-indexed: 11)
target_page = writer.pages[11]

# Create the link annotation
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

# Add annotation to page 7
if "/Annots" in page7:
    page7[NameObject("/Annots")].append(link_annotation)
else:
    page7[NameObject("/Annots")] = ArrayObject([link_annotation])

writer.write(OUTPUT_PDF)
print(f"Output PDF created: {OUTPUT_PDF}")

# Cleanup temp file
os.remove(TEMP_PDF)
print("Done!")

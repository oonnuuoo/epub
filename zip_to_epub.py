import os
import sys
import zipfile
import shutil
from PIL import Image, ImageDraw, ImageFont
from ebooklib import epub
import uuid


def create_cover_image(text, output_path, width=800, height=1200):
    """
    Create a cover image with the given text

    Args:
        text: Text to display on the cover
        output_path: Path to save the cover image
        width: Width of the cover image
        height: Height of the cover image
    """
    # Create a new image with white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Try to use a Japanese font, fallback to default if not available
    try:
        # Try common Japanese font paths on Windows
        font_paths = [
            'C:\\Windows\\Fonts\\msgothic.ttc',  # MS Gothic
            'C:\\Windows\\Fonts\\meiryo.ttc',    # Meiryo
            'C:\\Windows\\Fonts\\msmincho.ttc',  # MS Mincho
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 60)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Calculate text position (center of image)
    # For multiline text, split and calculate each line
    lines = []
    max_width = width - 100  # Leave 50px margin on each side

    # Simple word wrapping
    words = text.split()
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    # If no lines, just use the original text
    if not lines:
        lines = [text]

    # Calculate total height and starting Y position
    line_height = 80
    total_height = len(lines) * line_height
    y = (height - total_height) / 2

    # Draw each line centered
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) / 2
        draw.text((x, y), line, fill='black', font=font)
        y += line_height

    # Save the image
    img.save(output_path)


def unzip_file(zip_path):
    """
    Extract zip file to a directory with the same name

    Args:
        zip_path: Path to the zip file

    Returns:
        Path to the extracted directory
    """
    # Get directory name (zip file name without extension)
    dir_name = os.path.splitext(zip_path)[0]

    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dir_name)

    return dir_name


def get_sorted_images(directory):
    """
    Get all image files in directory sorted by filename

    Args:
        directory: Path to the directory

    Returns:
        List of image file paths sorted by filename
    """
    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

    # Get all image files
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(image_extensions):
            images.append(os.path.join(directory, filename))

    # Sort by filename
    images.sort()

    return images


def create_epub(title, author, publisher, cover_image_path, image_paths, output_path):
    """
    Create an EPUB file from images

    Args:
        title: Book title
        author: Author name
        publisher: Publisher name
        cover_image_path: Path to the cover image
        image_paths: List of image file paths
        output_path: Path to save the EPUB file
    """
    # Create EPUB book
    book = epub.EpubBook()

    # Set metadata
    book.set_identifier(str(uuid.uuid4()))
    book.set_title(title)
    book.set_language('ja')
    book.add_author(author)
    book.add_metadata('DC', 'publisher', publisher)

    # Set right-to-left page progression (right binding)
    book.set_direction('rtl')

    # Add cover image
    with open(cover_image_path, 'rb') as f:
        cover_image_data = f.read()

    cover_item = epub.EpubItem(
        uid="cover_image",
        file_name="images/cover.jpg",
        media_type="image/jpeg",
        content=cover_image_data
    )
    book.add_item(cover_item)

    # Create cover page
    cover_html = epub.EpubHtml(
        title='Cover',
        file_name='cover.xhtml',
        lang='ja'
    )
    cover_html.content = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
    <title>Cover</title>
    <style>
        body { margin: 0; padding: 0; text-align: center; }
        img { max-width: 100%; max-height: 100%; }
    </style>
</head>
<body>
    <div>
        <img src="images/cover.jpg" alt="Cover"/>
    </div>
</body>
</html>'''.encode('utf-8')
    book.add_item(cover_html)

    # Add all images
    chapters = [cover_html]
    spine_items = ['nav', cover_html]

    for idx, image_path in enumerate(image_paths, start=1):
        # Read image
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # Determine image type
        image_ext = os.path.splitext(image_path)[1].lower()
        if image_ext in ['.jpg', '.jpeg']:
            media_type = 'image/jpeg'
            file_ext = 'jpg'
        elif image_ext == '.png':
            media_type = 'image/png'
            file_ext = 'png'
        elif image_ext == '.gif':
            media_type = 'image/gif'
            file_ext = 'gif'
        else:
            # Default to JPEG
            media_type = 'image/jpeg'
            file_ext = 'jpg'

        # Add image item
        image_item = epub.EpubItem(
            uid=f"image_{idx}",
            file_name=f"images/page_{idx:04d}.{file_ext}",
            media_type=media_type,
            content=image_data
        )
        book.add_item(image_item)

        # Create HTML page for the image
        page = epub.EpubHtml(
            title=f'Page {idx}',
            file_name=f'page_{idx:04d}.xhtml',
            lang='ja'
        )
        page.content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
    <title>Page {idx}</title>
    <style>
        body {{ margin: 0; padding: 0; text-align: center; }}
        img {{ max-width: 100%; max-height: 100%; }}
    </style>
</head>
<body>
    <div>
        <img src="images/page_{idx:04d}.{file_ext}" alt="Page {idx}"/>
    </div>
</body>
</html>'''.encode('utf-8')
        book.add_item(page)
        chapters.append(page)
        spine_items.append(page)

    # Add navigation
    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Set spine (reading order)
    book.spine = spine_items

    # Write EPUB file
    epub.write_epub(output_path, book)


def main():
    """Main function"""
    print("=" * 60)
    print("ZIP to EPUB Converter")
    print("=" * 60)

    # Input zip file path
    zip_file_path = input("\nInput zip file path: ").strip()

    # Remove quote characters
    zip_file_path = zip_file_path.strip('"\'')

    # Check if file exists
    if not os.path.exists(zip_file_path):
        print(f"\nError: File not found: {zip_file_path}")
        return

    # Get book title from zip file name (without extension)
    title = os.path.splitext(os.path.basename(zip_file_path))[0]
    author = "hogehoge"
    publisher = "hogehoge"

    print(f"\nBook title: {title}")
    print(f"Author: {author}")
    print(f"Publisher: {publisher}")

    # Extract zip file
    print("\nExtracting zip file...")
    extracted_dir = unzip_file(zip_file_path)
    print(f"Extracted to: {extracted_dir}")

    # Get sorted images
    print("\nGetting images...")
    image_paths = get_sorted_images(extracted_dir)
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("\nError: No images found in zip file")
        shutil.rmtree(extracted_dir)
        return

    # Create cover image
    print("\nCreating cover image...")
    cover_image_path = os.path.join(extracted_dir, "cover.jpg")
    create_cover_image(title, cover_image_path)
    print(f"Cover image created: {cover_image_path}")

    # Create EPUB file
    epub_output_path = os.path.splitext(zip_file_path)[0] + ".epub"
    print(f"\nCreating EPUB file: {epub_output_path}")
    create_epub(title, author, publisher, cover_image_path, image_paths, epub_output_path)
    print(f"\nEPUB file created successfully: {epub_output_path}")

    # Clean up extracted directory
    print("\nCleaning up temporary files...")
    shutil.rmtree(extracted_dir)
    print("Done!")

    print("=" * 60)


if __name__ == "__main__":
    main()

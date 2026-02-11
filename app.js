// Global variables
let selectedFile = null;

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const titleInput = document.getElementById('titleInput');
const authorInput = document.getElementById('authorInput');
const publisherInput = document.getElementById('publisherInput');
const convertBtn = document.getElementById('convertBtn');
const progress = document.getElementById('progress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const errorDiv = document.getElementById('error');
const successDiv = document.getElementById('success');

// Event listeners
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
convertBtn.addEventListener('click', convertToEpub);

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && file.name.endsWith('.zip')) {
        selectedFile = file;
        displayFileInfo(file);
        convertBtn.disabled = false;
    } else {
        showError('ZIPファイルを選択してください');
    }
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

// Handle drag leave
function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

// Handle drop
function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.zip')) {
        selectedFile = file;
        fileInput.files = e.dataTransfer.files;
        displayFileInfo(file);
        convertBtn.disabled = false;
    } else {
        showError('ZIPファイルをドロップしてください');
    }
}

// Display file info
function displayFileInfo(file) {
    fileName.textContent = `📄 ${file.name}`;
    fileSize.textContent = `サイズ: ${formatFileSize(file.size)}`;
    fileInfo.classList.add('show');

    // Auto-fill title from filename
    const title = file.name.replace('.zip', '');
    titleInput.value = title;

    // Hide error and success messages
    errorDiv.classList.remove('show');
    successDiv.classList.remove('show');
}

// Format file size
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

// Show error
function showError(message) {
    errorDiv.textContent = '❌ ' + message;
    errorDiv.classList.add('show');
    successDiv.classList.remove('show');
}

// Show success
function showSuccess(message, downloadUrl, filename) {
    successDiv.innerHTML = `
        ✅ ${message}
        <br>
        <a href="${downloadUrl}" download="${filename}" class="download-btn">📥 EPUBをダウンロード</a>
    `;
    successDiv.classList.add('show');
    errorDiv.classList.remove('show');
}

// Update progress
function updateProgress(percent, text) {
    progressFill.style.width = percent + '%';
    progressFill.textContent = percent + '%';
    progressText.textContent = text;
}

// Create cover image
async function createCoverImage(text, width = 800, height = 1200) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    // White background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);

    // Text settings
    ctx.fillStyle = 'black';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Try different font sizes to fit text
    let fontSize = 60;
    ctx.font = `${fontSize}px "MS Gothic", "Meiryo", "MS Mincho", sans-serif`;

    // Word wrap
    const maxWidth = width - 100;
    const words = text.split('');
    const lines = [];
    let currentLine = '';

    for (const char of words) {
        const testLine = currentLine + char;
        const metrics = ctx.measureText(testLine);

        if (metrics.width > maxWidth && currentLine !== '') {
            lines.push(currentLine);
            currentLine = char;
        } else {
            currentLine = testLine;
        }
    }
    if (currentLine) {
        lines.push(currentLine);
    }

    // Draw text
    const lineHeight = 80;
    const totalHeight = lines.length * lineHeight;
    let y = (height - totalHeight) / 2 + lineHeight / 2;

    for (const line of lines) {
        ctx.fillText(line, width / 2, y);
        y += lineHeight;
    }

    // Convert to blob
    return new Promise((resolve) => {
        canvas.toBlob(resolve, 'image/jpeg', 0.95);
    });
}

// Get sorted image files from zip
async function getSortedImages(zip) {
    const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'];
    const images = [];

    for (const [filename, file] of Object.entries(zip.files)) {
        if (!file.dir) {
            const ext = filename.toLowerCase().substring(filename.lastIndexOf('.'));
            if (imageExtensions.includes(ext)) {
                images.push({
                    filename: filename,
                    file: file
                });
            }
        }
    }

    // Sort by filename
    images.sort((a, b) => a.filename.localeCompare(b.filename));

    return images;
}

// Create EPUB file
async function createEpub(title, author, publisher, coverBlob, images, direction = 'rtl') {
    const epub = new JSZip();

    // Generate UUID
    const uuid = 'urn:uuid:' + generateUUID();

    // Create mimetype file (must be first, uncompressed)
    epub.file('mimetype', 'application/epub+zip', { compression: 'STORE' });

    // Create META-INF/container.xml
    const containerXml = `<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>`;
    epub.file('META-INF/container.xml', containerXml);

    // Add cover image
    epub.file('OEBPS/images/cover.jpg', coverBlob);

    // Add all images
    const manifest = [];
    const spine = [];

    manifest.push(`    <item id="cover_image" href="images/cover.jpg" media-type="image/jpeg"/>`);
    manifest.push(`    <item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>`);
    manifest.push(`    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>`);

    spine.push(`    <itemref idref="cover"/>`);

    for (let i = 0; i < images.length; i++) {
        const idx = i + 1;
        const image = images[i];
        const ext = image.filename.toLowerCase().substring(image.filename.lastIndexOf('.'));
        const fileExt = ext === '.png' ? 'png' : 'jpg';
        const mediaType = ext === '.png' ? 'image/png' : 'image/jpeg';

        // Add image
        const imageData = await image.file.async('blob');
        epub.file(`OEBPS/images/page_${idx.toString().padStart(4, '0')}.${fileExt}`, imageData);

        // Add to manifest
        manifest.push(`    <item id="image_${idx}" href="images/page_${idx.toString().padStart(4, '0')}.${fileExt}" media-type="${mediaType}"/>`);
        manifest.push(`    <item id="page_${idx}" href="page_${idx.toString().padStart(4, '0')}.xhtml" media-type="application/xhtml+xml"/>`);

        // Add to spine
        spine.push(`    <itemref idref="page_${idx}"/>`);

        // Create HTML page
        const pageHtml = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
    <title>Page ${idx}</title>
    <style>
        body { margin: 0; padding: 0; text-align: center; }
        img { max-width: 100%; max-height: 100%; }
    </style>
</head>
<body>
    <div>
        <img src="images/page_${idx.toString().padStart(4, '0')}.${fileExt}" alt="Page ${idx}"/>
    </div>
</body>
</html>`;
        epub.file(`OEBPS/page_${idx.toString().padStart(4, '0')}.xhtml`, pageHtml);
    }

    // Create cover.xhtml
    const coverHtml = `<?xml version="1.0" encoding="UTF-8"?>
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
</html>`;
    epub.file('OEBPS/cover.xhtml', coverHtml);

    // Create content.opf
    const contentOpf = `<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid" xml:lang="ja" prefix="rendition: http://www.idpf.org/vocab/rendition/#">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:identifier id="uid">${uuid}</dc:identifier>
        <dc:title>${escapeXml(title)}</dc:title>
        <dc:language>ja</dc:language>
        <dc:creator>${escapeXml(author)}</dc:creator>
        <dc:publisher>${escapeXml(publisher)}</dc:publisher>
        <meta property="dcterms:modified">${new Date().toISOString().split('.')[0]}Z</meta>
        <meta property="rendition:layout">pre-paginated</meta>
        <meta property="rendition:spread">none</meta>
        <meta name="cover" content="cover_image"/>
    </metadata>
    <manifest>
${manifest.join('\n')}
    </manifest>
    <spine page-progression-direction="${direction}" toc="ncx">
${spine.join('\n')}
    </spine>
</package>`;
    epub.file('OEBPS/content.opf', contentOpf);

    // Create toc.ncx
    const navPoints = [];
    navPoints.push(`        <navPoint id="cover" playOrder="1">
            <navLabel><text>Cover</text></navLabel>
            <content src="cover.xhtml"/>
        </navPoint>`);

    for (let i = 0; i < images.length; i++) {
        const idx = i + 1;
        navPoints.push(`        <navPoint id="page_${idx}" playOrder="${idx + 1}">
            <navLabel><text>Page ${idx}</text></navLabel>
            <content src="page_${idx.toString().padStart(4, '0')}.xhtml"/>
        </navPoint>`);
    }

    const tocNcx = `<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
    <head>
        <meta name="dtb:uid" content="${uuid}"/>
        <meta name="dtb:depth" content="1"/>
        <meta name="dtb:totalPageCount" content="0"/>
        <meta name="dtb:maxPageNumber" content="0"/>
    </head>
    <docTitle>
        <text>${escapeXml(title)}</text>
    </docTitle>
    <navMap>
${navPoints.join('\n')}
    </navMap>
</ncx>`;
    epub.file('OEBPS/toc.ncx', tocNcx);

    return epub;
}

// Generate UUID
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Escape XML
function escapeXml(str) {
    return str.replace(/&/g, '&amp;')
              .replace(/</g, '&lt;')
              .replace(/>/g, '&gt;')
              .replace(/"/g, '&quot;')
              .replace(/'/g, '&apos;');
}

// Convert to EPUB
async function convertToEpub() {
    if (!selectedFile) {
        showError('ファイルが選択されていません');
        return;
    }

    try {
        // Reset UI
        errorDiv.classList.remove('show');
        successDiv.classList.remove('show');
        progress.classList.add('show');
        convertBtn.disabled = true;

        // Get metadata
        const title = titleInput.value || selectedFile.name.replace('.zip', '');
        const author = authorInput.value || 'hogehoge';
        const publisher = publisherInput.value || 'hogehoge';

        // Get direction from radio buttons
        const directionRadio = document.querySelector('input[name="direction"]:checked');
        const direction = directionRadio ? directionRadio.value : 'rtl';

        // Load ZIP file
        updateProgress(10, 'ZIPファイルを読み込み中...');
        const zip = await JSZip.loadAsync(selectedFile);

        // Get sorted images
        updateProgress(20, '画像を検索中...');
        const images = await getSortedImages(zip);

        if (images.length === 0) {
            throw new Error('ZIPファイル内に画像が見つかりませんでした');
        }

        // Create cover image
        updateProgress(30, 'カバー画像を生成中...');
        const coverBlob = await createCoverImage(title);

        // Create EPUB
        updateProgress(50, 'EPUBファイルを生成中...');
        const epub = await createEpub(title, author, publisher, coverBlob, images, direction);

        // Generate EPUB file
        updateProgress(80, 'EPUBファイルを保存中...');
        const epubBlob = await epub.generateAsync({
            type: 'blob',
            mimeType: 'application/epub+zip',
            compression: 'DEFLATE',
            compressionOptions: { level: 9 }
        });

        // Create download link
        updateProgress(100, '完了!');
        const downloadUrl = URL.createObjectURL(epubBlob);
        const filename = title + '.epub';

        setTimeout(() => {
            progress.classList.remove('show');
            showSuccess('EPUBファイルの生成が完了しました!', downloadUrl, filename);
            convertBtn.disabled = false;
        }, 500);

    } catch (error) {
        console.error(error);
        progress.classList.remove('show');
        showError('変換中にエラーが発生しました: ' + error.message);
        convertBtn.disabled = false;
    }
}

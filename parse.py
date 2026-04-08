import csv
import pandas as pd
import sys
import os
import re
import zipfile

args = sys.argv

epub_file = '../../' + args[1] + '/' + args[1] + '.epub'
zip_file = '../../' + args[1] + '/' + args[1] + '.zip'

filename = '../../' + args[1] + '/index.csv'
coods_file = '../../' + args[1] + '/ocr/output.csv'
target = '../../' + args[1] + '/xhtml.txt'
output_file = open('../../' + args[1] + '/xhtml.txt', 'w', encoding='utf_8_sig')
output_file2 = open('../../' + args[1] + '/xhtml2.txt', 'w', encoding='utf_8_sig')
output_file3 = open('../../' + args[1] + '/xhtml3.txt', 'w', encoding='utf_8_sig')
output_file4 = open('../../' + args[1] + '/xhtml4.txt', 'w', encoding='utf_8_sig')

def unzip_zip_file(zip_file_path):
    # unzip zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(zip_file_path.replace('.zip', ''))

    return


os.rename(epub_file, zip_file)

unzip_zip_file(zip_file)


with open(filename, encoding='utf_8_sig', newline='') as input_file:
    # reader = csv.reader(input_file)
    # for row in reader:
    #     num = str(row[0])
    #     index = str(row[2])
    #     hierarchy = row[1]
    #     line = '<li><a href="xhtml/p-' + num + '.xhtml">' + index + '</a>\n'
    #     output_file.write(line)
    #     next_line = next(reader)
    #     if int(next_line[1]) == int(hierarchy)-2:
    #         output_file.write('</li>\n</ol>\n</li>\n</ol>\n</li>\n')
    #     elif int(next_line[1]) == int(hierarchy)-1:
    #         output_file.write('</li>\n</ol>\n</li>\n')
    #     elif int(next_line[1]) == int(hierarchy):
    #         output_file.write('</li>\n')
    #     else:
    #         output_file.write('<ol style="list-style-type: none;">\n')

    reader = csv.reader(input_file)
    data = [line for line in reader]
    
    def safe_int(value, default=0):
        try:
            text = str(value).strip()
            return int(text) if text != '' else default
        except (ValueError, TypeError):
            return default
    lines = len(data)
    for i in range(lines-1):

        num = str(data[i][0]).zfill(4)          # ページ通し番号(4digits)
        index = str(data[i][2])                 # タイトル
        cur_h = safe_int(data[i][1], 0)         # currentのhierarchy
        next_h = safe_int(data[i+1][1], cur_h)  # next hierarchy
        line = '<li><a href="xhtml/p-' + num + '.xhtml">' + index + '</a>' # 各行の出力文字列
        output_file.write(line)
        if next_h == cur_h - 4:
            output_file.write('</li>\n</ol>\n</li>\n</ol>\n</li>\n</ol>\n</li>\n</ol>\n</li>\n')
        elif next_h == cur_h - 3:
        # if int(data[i+1][1]) == int(data[i][1]) - 3:
            output_file.write('</li>\n</ol>\n</li>\n</ol>\n</li>\n</ol>\n</li>\n')
        elif next_h == cur_h - 2:
            output_file.write('</li>\n</ol>\n</li>\n</ol>\n</li>\n')
        elif next_h == cur_h - 1:
            output_file.write('</li>\n</ol>\n</li>\n')
        elif next_h == cur_h:
            output_file.write('</li>\n')
        else:
            output_file.write('\n<ol style="list-style-type: none;">\n')
    
    print(str(lines) + str(data[lines-1][1]))
    
    if str(data[lines-1][1]) == "3":
        print("lines=" + str(lines) + "data = " + str(data[lines-1][1]))
        output_file.write('<li><a href="xhtml/p-' + str(data[len(data)-1][0]) + '.xhtml">' + str(data[len(data)-1][2]) + '</a></li>\n</ol>\n</li>\n</ol>\n</li>')
    elif str(data[lines-1][1]) == "2":
        print("lines=" + str(lines) + "data = " + str(data[lines-1][1]))
        output_file.write('<li><a href="xhtml/p-' + str(data[len(data)-1][0]) + '.xhtml">' + str(data[len(data)-1][2]) + '</a></li>\n</ol>\n</li>')
    else:
        print("lines=" + str(lines) + "data = " + str(data[lines-1][1]))
        output_file.write('<li><a href="xhtml/p-' + str(data[len(data)-1][0]) + '.xhtml">' + str(data[len(data)-1][2]) + '</a></li>')



    for i in range(lines):
        output_file2.write('<li><a href="xhtml/p-' + str(data[i][0]).zfill(4) + '.xhtml#pagenum_' + str(i) + '">' + str(data[i][3]) + '</a></li>\n')

    for i in range(lines):
        output_file3.write('<p class="indent-000' + str(safe_int(data[i][1], 0)) + '">\n<a href="p-' + str(data[i][0]).zfill(4) + '.xhtml">' + str(data[i][2]) + '</a></p>\n')

output_file.close()
output_file2.close()
output_file3.close()

def modify_navigation_xhtml():
    nav_path = '../../' + args[1] + '/' + args[1] + '/item/navigation-documents.xhtml'
    xhtml_path = '../../' + args[1] + '/xhtml.txt'
    xhtml2_path = '../../' + args[1] + '/xhtml2.txt'

    with open(nav_path, 'r', encoding='utf_8_sig') as f:
        content = f.read()

    with open(xhtml_path, 'r', encoding='utf_8_sig') as f:
        xhtml_content = f.read()

    with open(xhtml2_path, 'r', encoding='utf_8_sig') as f:
        xhtml2_content = f.read()

    def find_nav_ol_range(text, nav_id):
        """Find start/end positions of the content inside the main <ol>...</ol> of nav with given id."""
        nav_match = re.search(r'<nav[^>]*id="' + nav_id + r'"[^>]*>', text)
        if not nav_match:
            return None
        ol_match = re.search(r'<ol[^>]*>', text[nav_match.end():])
        if not ol_match:
            return None
        ol_content_start = nav_match.end() + ol_match.end()
        # Find matching </ol> by tracking depth
        depth = 1
        pos = ol_content_start
        while pos < len(text) and depth > 0:
            ol_open = re.search(r'<ol[\s>]', text[pos:])
            ol_close = re.search(r'</ol>', text[pos:])
            if ol_close is None:
                break
            if ol_open and ol_open.start() < ol_close.start():
                depth += 1
                pos = pos + ol_open.end()
            else:
                depth -= 1
                if depth == 0:
                    return (ol_content_start, pos + ol_close.start())
                pos = pos + ol_close.end()
        return None

    def find_nth_toplevel_li_end(text, n):
        """Find position after the nth top-level </li> in text."""
        depth = 0
        count = 0
        tag_pattern = re.compile(r'<li[\s>/]|</li>')
        pos = 0
        while pos < len(text):
            m = tag_pattern.search(text, pos)
            if not m:
                break
            if m.group().startswith('<li'):
                depth += 1
                pos = m.end()
            else:  # </li>
                depth -= 1
                pos = m.end()
                if depth == 0:
                    count += 1
                    if count == n:
                        return pos
        return -1

    # --- Process nav#toc: keep first 3 <li>, replace the rest with xhtml.txt ---
    toc_range = find_nav_ol_range(content, 'toc')
    if toc_range:
        ol_start, ol_end = toc_range
        ol_content = content[ol_start:ol_end]
        pos = find_nth_toplevel_li_end(ol_content, 2)
        if pos != -1:
            kept = ol_content[:pos]
            content = content[:ol_start] + kept + '\n' + xhtml_content + '\n' + content[ol_end:]

    # --- Process nav#page-list: replace all <li> with xhtml2.txt ---
    pl_range = find_nav_ol_range(content, 'page-list')
    if pl_range:
        ol_start, ol_end = pl_range
        content = content[:ol_start] + '\n' + xhtml2_content + '\n' + content[ol_end:]

    with open(nav_path, 'w', encoding='utf_8_sig') as f:
        f.write(content)

modify_navigation_xhtml()


def modify_toc_xhtml():
    toc_path = '../../' + args[1] + '/' + args[1] + '/item/xhtml/p-0001.xhtml'
    xhtml3_path = '../../' + args[1] + '/xhtml3.txt'

    with open(toc_path, 'r', encoding='utf_8_sig') as f:
        content = f.read()

    with open(xhtml3_path, 'r', encoding='utf_8_sig') as f:
        xhtml3_content = f.read()

    # Find the range covering all <p>...</p> elements and replace with xhtml3.txt
    first_p = re.search(r'<p[\s>]', content)
    last_p_end = None
    for m in re.finditer(r'</p>', content):
        last_p_end = m.end()

    if first_p and last_p_end:
        content = content[:first_p.start()] + xhtml3_content + '\n' + content[last_p_end:]

    with open(toc_path, 'w', encoding='utf_8_sig') as f:
        f.write(content)

modify_toc_xhtml()


def replace_style_directory():
    src = '../style'
    dst = '../../' + args[1] + '/' + args[1] + '/item/style'
    if os.path.exists(dst):
        import shutil
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

replace_style_directory()

os.rename(zip_file, epub_file)

with open(coods_file, 'r', encoding='utf_8_sig') as f:
    # <a xlink:href="p-0004.xhtml">
    #  <rect x="112" y="412" width="260" height="49" fill="#000000" fill-opacity="0" opacity="0" pointer-events="all"/>
    # </a>
    # x:row4 row y:row5 width:row6 height:row7 link: row10
    
    reader =csv.reader(f)
    data = [line for line in reader]

    lines = len(data)

    img_name = data[0][0]
    output_file4.write('for ' + img_name + ':\n\n\n')

    for i in range(lines-1):
        x = data[i][3]
        y = data[i][4]
        width = data[i][5]
        height = data[i][6]
        link =str(data[i][9]).zfill(4)

        output_line = '<a xlink:href="p-' + link + '.xhtml">\n' + '<rect x="' + x + '" y="' + y + '" width="' + width + '" height="' + height + '" fill="#000000" fill-opacity="0" opacity="0" pointer-events="all"/>\n' + '</a>'
        output_file4.write(output_line)

        if data[i+1][0] != img_name:
            img_name = data[i+1][0]
            output_file4.write('\n\n\n' + 'for ' + img_name + ':\n\n\n')
        
        else:
            output_file4.write('\n')

    output_file4.write('<a xlink:href="p-' + str(data[len(data)-1][0]).zfill(4) + '.xhtml">\n' + '<rect x="' + str(data[len(data)-1][3]) + '" y="' + str(data[len(data)-1][4]) + '" width="' + str(data[len(data)-1][5]) + '" height="' + str(data[len(data)-1][6]) + '" fill="#000000" fill-opacity="0" opacity="0" pointer-events="all"/>\n' + '</a>')

output_file4.close()


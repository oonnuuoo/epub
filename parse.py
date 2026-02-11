import csv
import pandas as pd
import sys
import os
import zipfile

args = sys.argv

epub_file = '../../' + args[1] + '/' + args[1] + '.epub'
zip_file = '../../' + args[1] + '/' + args[1] + '.zip'

filename = '../../' + args[1] + '/index.csv'
target = '../../' + args[1] + '/xhtml.txt'
output_file = open('../../' + args[1] + '/xhtml.txt', 'w', encoding='utf-8-sig')
output_file2 = open('../../' + args[1] + '/xhtml2.txt', 'w', encoding='utf-8-sig')
output_file3 = open('../../' + args[1] + '/xhtml3.txt', 'w', encoding='utf-8-sig')


def unzip_zip_file(zip_file_path):
    # unzip zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(zip_file_path.replace('.zip', ''))

    return


os.rename(epub_file, zip_file)

unzip_zip_file(zip_file)


with open(filename, encoding='utf-8-sig', newline='') as input_file:
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
        num = str(data[i][0])                   # ページ通し番号
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
        output_file2.write('<li><a href="xhtml/p-' + str(data[i][0]) + '.xhtml#pagenum_' + str(i) + '">' + str(data[i][3]) + '</a></li>\n')

    for i in range(lines):
        output_file3.write('<p class="indent-000' + str(safe_int(data[i][1], 0)) + '">\n<a href="p-' + str(data[i][0]) + '.xhtml">' + str(data[i][2]) + '</a></p>\n')

output_file.close()
output_file2.close()
output_file3.close()

# target_xhtml_path = '../../' + args[1] + '/' + args[1] + '/item/navigation-documents.xhtml'
# target_file = open(target_xhtml_path, encoding='utf-8-sig')

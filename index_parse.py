# import OCR result .xml file with text and its displacement in the original image, 
# each line formatted as:
# <LINE TYPE="本文" X="{x}" Y="{y}" WIDTH="{w}" HEIGHT="{h}" CONF="0.909" PRED_CHAR_CNT="1.000" ORDER="2" STRING="{index_title}"/>
#  and parse them, then generate csv file with the following columns:
# [img_name, x, y, w, h, index_title, page_number]
# page_number is the last part of the index_title

import csv
import pandas as pd
import sys
import os
import re

args = sys.argv

# input_xml = '../../' + 'ocrtest/result' + args[1] + '.xml'
# output_csv = '../../' + 'ocrtest/result' + args[1] + '.csv'

# input_xml = '../../' + args[1] + '/ocr/test.xml'
input_dir = '../../' + args[1] + '/ocr'
output_csv = '../../' + args[1] + '/ocr/output.csv'

# format of input xml
# <LINE TYPE="本文" X="65" Y="381" WIDTH="2140" HEIGHT="56" CONF="0.909" PRED_CHAR_CNT="1.000" ORDER="2" STRING="1.1都市の概念と定義…………………………………………………………………………………………………!"/>
# parse the xml file and generate the csv file



def parse_xml(input_xml):
    output_file = open(output_csv, 'a', encoding='utf-8-sig')
    with open(input_xml, 'r', encoding='utf-8') as input_file:

    # pattern = r'X="(?P<x_dis>.*?)" Y="(?P<y_dis>.*?)" WIDTH="(?P<width>.*?)" HEIGHT="(?P<height>.*?)" CONF="\d+\.\d+" PRED_CHAR_CNT="\d+\.\d+" ORDER="[0-9]{1,3}" STRING="(?P<index>.*?)"/>'

    # get image name and size
    # <PAGE IMAGENAME="000_007.jpg" WIDTH="2287" HEIGHT="3195">


    # pattern1 = r'IMAGENAME="(?P<img_name>.*+)"\.jpg"\s+WIDTH=""(?P<img_width>\d+)""\s+HEIGHT=""(?P<img_height>\d+)"">'
    
    # for line in input_file:
    #     match1 = re.search(pattern1, line)
    #     if match1:
    #         img_name = match1.group('img_name')
    #         img_width = match1.group('img_width')
    #         img_height = match1.group('img_height')
    #         print(img_name, img_width, img_height)
    #     else:
    #         print("skip")
        pattern1 = r'IMAGENAME="(?P<IMAGENAME>[^"]+)"\s+WIDTH="(?P<IMGWIDTH>\d+)"\s+HEIGHT="(?P<IMGHEIGHT>\d+)"'
        pattern2 = r'X="(?P<X>\d+)"\s+Y="(?P<Y>\d+)"\s+WIDTH="(?P<WIDTH>\d+)"\s+HEIGHT="(?P<HEIGHT>\d+)".*?STRING="(?P<STRING>[^"]*)"'
        for line in input_file:
            line = line.replace(',', '')
            match1 = re.search(pattern1, line)
            match2 = re.search(pattern2, line)
            if match1:
                img_name = match1.group('IMAGENAME')
                img_width = match1.group('IMGWIDTH')
                img_height = match1.group('IMGHEIGHT')
                print(img_name, img_width, img_height)
                continue
            elif match2:
                x = match2.group('X')
                y = match2.group('Y')
                width = match2.group('WIDTH')
                height = match2.group('HEIGHT')
                string = match2.group('STRING')
                match3 = re.search(r'\d+$', string)
                if match3:
                    page_num = match3.group(0)
                else:
                    page_num = "0"

                print(x, y, width, height, string, page_num)
                output_file.write(img_name + ',' + img_width + ',' + img_height + ',' + x + ',' + y + ',' + width + ',' + height + ',' + string + ',' + page_num + '\n')
                continue
            else:
                print("skip")
                continue

file_list = os.listdir(input_dir)
for file_name in file_list:
    if file_name.endswith('.xml'):
        file_path = os.path.join(input_dir, file_name)
        parse_xml(file_path)
        print("parsing..." + file_path)
    else:
        print("else")






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

input_xml = '../../' + 'ocrtest/result' + args[1] + '.xml'
output_csv = '../../' + 'ocrtest/result' + args[1] + '.csv'


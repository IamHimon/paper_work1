# from lxml import etree
import xml.parsers.expat
from collections import defaultdict
import json

DISCOGS_DB_DUMP = '/home/himon/Jobs/paper_work1/dataset/discogs_20120301_masters.xml'

data = defaultdict(dict)
curr_id = 0
curr_tag = None
curr_value = ""
useful_tags = ['artist', 'name', 'genre', 'year', 'style', 'title', 'main_release']


def start_element(name, attrs):
    global curr_id, curr_tag, data
    if name == 'master':
        curr_id = attrs['id']
        data[curr_id] = defaultdict(list)
    if name in useful_tags:
        curr_tag = name
    else:
        curr_tag = None


def char_data(value):
    global curr_value
    if curr_tag and curr_id:
        curr_value = curr_value + value


def end_element(name):
    global curr_id, curr_tag, data, curr_value
    if curr_tag and curr_id and curr_value:
        data[curr_id][curr_tag].append(curr_value)
    if name == 'master':
        curr_id = 0
    curr_tag = None
    curr_value = ""

p = xml.parsers.expat.ParserCreate()
p.StartElementHandler = start_element
p.EndElementHandler = end_element
p.CharacterDataHandler = char_data

f = open(DISCOGS_DB_DUMP)
p.ParseFile(f)

print(data)

# w = open('data.json', 'w')
# json.dump(data, w)
import urllib
import codecs
import sys
import os
from xml.dom import minidom
import xml.etree.cElementTree as ET
import musicbrainzngs

# result = musicbrainzngs.get_releases_by_discid(disc.id, includes=["artists", "recordings"])


#
musicbrainzngs.set_useragent("Example music app", "0.1", "http://example.com/music")
# # artist_id = 'f4f645c0-1395-4a9e-9681-996cdad28ed3'
#
# artist_id = "84e9d30c-a910-4e2a-b7d8-ce034b1244e3"
# try:
#     result = musicbrainzngs.get_artist_by_id(artist_id)
#     print(result)
# except WebServiceError as exc:
#     print("Something went wrong with the request: %s" % exc)
# else:
#     artist = result["artist"]
#
#     print("name:\t\t%s" % artist["name"])
#     if 'type' in artist:
#         print("type:\t%s" % artist["type"])
#     if "country" in artist:
#         print("country:\t%s" % artist["country"])
#     if 'gender' in artist:
#         print("gender:\t%s" % artist["gender"])
# #
#
# result = musicbrainzngs.get_artist_by_id(artist_id,
#               includes=["release-groups"], release_type=["album", "ep"])
# print(result)
# for release_group in result["artist"]["release-group-list"]:
#     # print("{title}, {type}".format(title=release_group["title"], type=release_group["type"]))
#     print('title: %s' % (release_group['title']))
#     print('first-release-date:', release_group['first-release-date'])


# artist_id = "2f99de9d-8ab2-44ff-b20c-f55ccb35e61e"
# result = musicbrainzngs.get_artist_by_id(artist_id)
# print(result)
# artist = result["artist"]
# print(artist)
# print(artist["life-span"])
# print(artist["country"])
# print(artist["begin-area"])
# print(artist["type"])
# print(artist["sort-name"])





label = "71247f6b-fd24-4a56-89a2-23512f006f0c"
limit = 100
offset = 0
releases = []
page = 1
print("fetching page number %d.." % page)
result = musicbrainzngs.browse_releases(label=label, includes=["labels"],
                release_type=["album"], limit=limit)


page_releases = result['release-list']
print(page_releases)

releases += page_releases

# release-count is only available starting with musicbrainzngs 0.5
if "release-count" in result:
        count = result['release-count']
        print("")
while len(page_releases) >= limit:
    offset += limit
    page += 1
    print("fetching page number %d.." % page)
    result = musicbrainzngs.browse_releases(label=label, includes=["labels"],
                        release_type=["album"], limit=limit, offset=offset)
    page_releases = result['release-list']
    releases += page_releases

print("")
for release in releases:
    for label_info in release['label-info-list']:
        catnum = label_info.get('catalog-number')
        if label_info['label']['id'] == label and catnum:
            print("{catnum:>17}: {date:10} {title}".format(catnum=catnum,
                        date=release['date'], title=release['title']))
print("\n%d releases on  %d pages" % (len(releases), page))

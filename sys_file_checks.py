#!/usr/bin/python -tt
"""
Created on Fri Jan 05 15:45:32 2018

To do:
    Remove dupes?
    Find and rotate images on side
    Select "best" images for printing

@author: adeacon
"""

import os
#import sys
import hashlib
from PIL import Image
from PIL.ExifTags import TAGS

dir_name = r'C:\Users\adeacon\picture'
dpi_min = 90 #300

def findDup(parentFolder):
    # Dups in format {hash:[names]}
    dups = {}
    for dirName, subdirs, fileList in os.walk(parentFolder):
        print(('Scanning %s...' % dirName))
        for filename in fileList:
            # Get the path to the file
            path = os.path.join(dirName, filename)
            # Calculate hash
            file_hash = hashfile(path)
            # Add or append the file path
            if file_hash in dups:
                dups[file_hash].append(path)
            else:
                dups[file_hash] = [path]
    return dups


# Joins two dictionaries
def joinDicts(dict1, dict2):
    for key in list(dict2.keys()):
        if key in dict1:
            dict1[key] = dict1[key] + dict2[key]
        else:
            dict1[key] = dict2[key]


def hashfile(path, blocksize = 65536):
    afile = open(path, 'rb')
    hasher = hashlib.md5()
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    afile.close()
    return hasher.hexdigest()


def printResults(dict1):
    results = list([x for x in list(dict1.values()) if len(x) > 1])
    if len(results) > 0:
        print('Duplicates Found:')
        print('The following files are identical. The name could differ, but the content is identical')
        print('___________________')
        for result in results:
            for subresult in result:
                print(('\t\t%s' % subresult))
            print('___________________')

    else:
        print('No duplicate files found.')

def find_all_dupes():
    dups = {}
    #folders = sys.argv[1:]
    #for i in dir_name:
    #print i
    # Iterate the folders given
    if os.path.exists(dir_name):
        # Find the duplicated files and append them to the dups
        print(dir_name)
        joinDicts(dups, findDup(dir_name))
    else:
        print(('%s is not a valid path, please verify' % dir_name))
        #sys.exit()
    printResults(dups)

def check_names():
    for root, dirs, files in os.walk(dir_name, topdown=False):
        for name in sorted(files):
            if name.endswith('jpg'):
                print(name)

def get_exif(fn):
    ret = {}
    i = Image.open(fn)
    info = i._getexif()
    try:
        for tag, value in list(info.items()):
            decoded = TAGS.get(tag, tag)
            ret[decoded] = value
    except:
        pass
    return ret

def get_image_metadata():
    image_counter = 0
    for root, dirs, files in os.walk(dir_name, topdown=False):
        for name in sorted(files):
            if name.endswith('jpg'):
                #print name
                file_name = os.path.join(root, name)
                #print file_name
                exif = get_exif(file_name)
                XResolution = exif.get("XResolution", (0,1))
                YResolution = exif.get("YResolution", (0,1))
                #print XResolution[0]/XResolution[1]
                if (XResolution[0]/XResolution[1] >= dpi_min) & (YResolution[0]/YResolution[1] >= dpi_min):
                    image_counter += 1
                    #print "######### PRINT IT! #########"
                    print(file_name)
                    print("Make/Model: "+exif.get("Make", "")+" / "+exif.get("Model", ""))
                    print("Resolution: "+str(exif.get("XResolution", ""))+" / "+str(exif.get("YResolution", "")))
                    print("Dimensions: "+str(exif.get("ExifImageWidth", ""))+" / "+str(exif.get("ExifImageHeight", "")))
                    print("Orientation: "+str(exif.get("Orientation", "")))
                    print("Date: "+exif.get("DateTimeOriginal", ""))
                    #for key, value in exif.iteritems():
                    #    print key,
                    #print exif
                    print("\n\n")
    print("Number of selected images: "+str(image_counter))

def main():
    # check_names()
    # find_all_dupes()
    get_image_metadata()

if __name__ == '__main__':
    main()

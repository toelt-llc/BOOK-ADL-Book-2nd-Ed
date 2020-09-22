import os, sys, random
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile
from six.moves import urllib

class read_data():

    def __init__(self, CACHE_DIR, url_base):
        # We'll use the following directory to store files we download as well as our
        # preprocessed dataset.
        self.CACHE_DIR = CACHE_DIR
        self.url_base = url_base
        self.annotations = glob('BCCD_Dataset/BCCD/Annotations/*.xml')

    def cache_or_download_file(self, cache_dir, url_base, filename):
        """Read a cached file or download it."""
        filepath = os.path.join(cache_dir, filename)
        if tf.io.gfile.exists(filepath):
          return filepath
        if not tf.io.gfile.exists(cache_dir):
          tf.io.gfile.makedirs(cache_dir)
        url = os.path.join(url_base, filename)
        print('Downloading {url} to {filepath}.'.format(url = url, filepath = filepath))
        urllib.request.urlretrieve(url, filepath)
        return filepath

    def preprocess_bccd_dataset(self):
        """Download and preprocess the bccd dataset, generating a csv file.

        ### Author/Developer: Nicolas CHEN
        ### Filename: export.py
        ### Version: 1.0
        ### Field of research: Deep Learning in medical imaging
        ### Purpose: This Python script creates the CSV file from XML files.
        ### Output: This Python script creates the file "test.csv"
        ### with all data needed: filename, class_name, x1,y1,x2,y2
        ### HISTORY
        ### Version | Date          | Author       | Evolution
        ### 1.0     | 17/11/2018    | Nicolas CHEN | Initial version

        """
        df = []
        cnt = 0
        for file in self.url_base + self.annotations:
            f = self.cache_or_download_file(self.CACHE_DIR, self.url_base, file)
            filename = file.split('/')[-1]
            filename = filename.split('.')[0] + '.jpg'
            row = []
            parsedXML = ET.parse(file)
            for node in parsedXML.getroot().iter('object'):
                blood_cells = node.find('name').text
                xmin = int(node.find('bndbox/xmin').text)
                xmax = int(node.find('bndbox/xmax').text)
                ymin = int(node.find('bndbox/ymin').text)
                ymax = int(node.find('bndbox/ymax').text)

                row = [filename, blood_cells, xmin, xmax, ymin, ymax]
                df.append(row)
                cnt += 1

        data = pd.DataFrame(df, columns = ['filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax'])

        return data

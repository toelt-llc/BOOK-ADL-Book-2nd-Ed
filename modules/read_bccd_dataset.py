import os, sys, random
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile

class read_data():

    def __init__(self):
        self.annotations = glob('BCCD/Annotations/*.xml')

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
        for file in self.annotations:
            filename = file.split('\\')[-1]
            filename =filename.split('.')[0] + '.jpg'
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

        data = pd.DataFrame(df, columns=['filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax'])

        return data







import os
import pandas as pd
import numpy as np
from six.moves import urllib

import tensorflow as tf



  def download_bccd_dataset(self):
    """Download the radon dataset and read as Pandas dataframe."""
    srrs2 = pd.read_csv(self.cache_or_download_file(self.CACHE_DIR, self.url_base, 'srrs2.dat'))
    srrs2.rename(columns = str.strip, inplace = True)
    cty = pd.read_csv(self.cache_or_download_file(self.CACHE_DIR, self.url_base, 'cty.dat'))
    cty.rename(columns = str.strip, inplace = True)
    return srrs2, cty


  def preprocess_bccd_dataset(self, srrs2, cty, state = 'MN'):
    """Preprocess radon dataset as done in "Bayesian Data Analysis" book."""
    srrs2 = srrs2[srrs2.state == state].copy()
    cty = cty[cty.st == state].copy()

    # We will now join datasets on Federal Information Processing Standards
    # (FIPS) id, ie, codes that link geographic units, counties and county
    # equivalents. http://jeffgill.org/Teaching/rpqm_9.pdf
    srrs2['fips'] = 1000 * srrs2.stfips + srrs2.cntyfips
    cty['fips'] = 1000 * cty.stfips + cty.ctfips

    df = srrs2.merge(cty[['fips', 'Uppm']], on='fips')
    df = df.drop_duplicates(subset = 'idnum')
    df = df.rename(index = str, columns = {'Uppm': 'uranium_ppm'})

    # For any missing or invalid activity readings, we'll use a value of `0.1`.
    df['radon'] = df.activity.apply(lambda x: x if x > 0. else 0.1)

    # Remap categories to start from 0 and end at max(category).
    county_name = sorted(df.county.unique())
    df['county'] = df.county.astype(
        pd.api.types.CategoricalDtype(categories = county_name)).cat.codes
    county_name = list(map(str.strip, county_name))

    df['log_uranium_ppm'] = df['uranium_ppm']
    df_features = df[['floor', 'county', 'log_uranium_ppm', 'pcterr']]
    df_labels = df['radon']

    return df_features, df_labels, county_name


  def create_dataset(self):
    """Return the final dataframe with preprocessed data."""
    radon_features, radon_labels, county_name = self.preprocess_radon_dataset(*self.download_radon_dataset())

    return radon_features, radon_labels, county_name

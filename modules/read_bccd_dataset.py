import os, sys, random
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile

class read_data():

    def __init__(self):
        self.annotations = glob('BCCD_Dataset/BCCD/Annotations/*.xml')

    def preprocess_bccd_dataset(self):
        """Download and preprocess the bccd dataset, generating a csv file.

        ### HISTORY
        ### Version | Date          | Author       | Evolution
        ### 1.0     | 17/11/2018    | Nicolas CHEN | Initial version

        """
        df = []
        cnt = 0
        for file in self.annotations:
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

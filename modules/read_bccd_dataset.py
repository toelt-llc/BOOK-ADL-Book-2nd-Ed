import os, sys, random
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile
import cv2

class read_data():

    def __init__(self):
        self.annotations_path = 'BCCD_Dataset/BCCD/Annotations/'
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

    def plot_bccd_dataset(self,imagename):
        """Generate labeled img like example.jpg from the bccd dataset."""
        image = cv2.imread('BCCD_Dataset/BCCD/JPEGImages/' + imagename)
        tree = ET.parse(self.annotations_path + imagename)
        for elem in tree.iter():
        	if 'object' in elem.tag or 'part' in elem.tag:
        		for attr in list(elem):
        			if 'name' in attr.tag:
        				name = attr.text
        			if 'bndbox' in attr.tag:
        				for dim in list(attr):
        					if 'xmin' in dim.tag:
        						xmin = int(round(float(dim.text)))
        					if 'ymin' in dim.tag:
        						ymin = int(round(float(dim.text)))
        					if 'xmax' in dim.tag:
        						xmax = int(round(float(dim.text)))
        					if 'ymax' in dim.tag:
        						ymax = int(round(float(dim.text)))
        				if name[0] == "R":
        					cv2.rectangle(image, (xmin, ymin),
        								(xmax, ymax), (0, 255, 0), 1)
        					cv2.putText(image, name, (xmin + 10, ymin + 15),
        							cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 255, 0), 1)
        				if name[0] == "W":
        					cv2.rectangle(image, (xmin, ymin),
        								(xmax, ymax), (0, 0, 255), 1)
        					cv2.putText(image, name, (xmin + 10, ymin + 15),
        							cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)
        				if name[0] == "P":
        					cv2.rectangle(image, (xmin, ymin),
        								(xmax, ymax), (255, 0, 0), 1)
        					cv2.putText(image, name, (xmin + 10, ymin + 15),
        							cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (255, 0, 0), 1)

        cv2.imshow("test", image)

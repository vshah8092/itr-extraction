import cv2
import numpy as np
import os
import json
import pandas as pd
import PIL
import pytesseract
import sys
import torch

from config.classes import class_list
from config.constants import PathConstants
from pdf2jpg import pdf2jpg
from PIL import Image

valid_extension_lists = ["jpg", "jpeg", "png", "tif", "pdf"]

#Check for invalid file extensions and
#Convert .pdf, .tif (if any) to .jpg
for file in os.listdir(PathConstants.DATASET_FILE_PATH):
    filename = os.path.basename(file)
    if filename.split('.')[-1].lower() not in valid_extension_lists:
        print("Error: Invalid file extension.")
        sys.exit()
    if filename.split('.')[-1].lower() == "pdf":
        pdf2jpg(PathConstants.DATASET_FILE_PATH + file, filename.split()[0])
        os.remove(PathConstants.DATASET_FILE_PATH + file)
    if filename.split('.')[-1].lower() == "tif":
        read = cv2.imread(PathConstants.DATASET_FILE_PATH + file)
        outfile = file.split('.')[0] + '.jpg'
        cv2.imwrite(PathConstants.DATASET_FILE_PATH + outfile, read)
        os.remove(PathConstants.DATASET_FILE_PATH + file)
print("All files are ready for processing.")

#Setup for processing, Clear existing output files
pytesseract.pytesseract.tesseract_cmd = PathConstants.TESSERACT_PATH
for file in os.listdir(PathConstants.ANNOTATED_IMAGE_PATH):
    os.remove(PathConstants.ANNOTATED_IMAGE_PATH + file)
for file in os.listdir(PathConstants.JSON_FILE_PATH):
    os.remove(PathConstants.JSON_FILE_PATH + file)
for file in os.listdir(PathConstants.OUTPUT_EXCEL):
    os.remove(PathConstants.OUTPUT_EXCEL + file)

#YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path = PathConstants.YOLO_MODEL_PATH)
model.conf = 0

count = 1
output_excel = pd.DataFrame()

#Processing of a single file
for file in os.listdir(PathConstants.DATASET_FILE_PATH):
    filename = os.path.basename(file)
    img = np.array(Image.open(PathConstants.DATASET_FILE_PATH + file))
    results = model(img)
    df = results.pandas().xyxy[0]
    req_values = {}
    bounding_boxes = []

    #Collecting bounding box with highest confidence value for each class
    for i in range(15):
        if i in df['class'].unique().tolist():
            xmin = df[df['class'] == i].sort_values(by = ['confidence'], ascending = False).iloc[0]['xmin']
            xmax = df[df['class'] == i].sort_values(by = ['confidence'], ascending = False).iloc[0]['xmax']
            ymin = df[df['class'] == i].sort_values(by = ['confidence'], ascending = False).iloc[0]['ymin']
            ymax = df[df['class'] == i].sort_values(by = ['confidence'], ascending = False).iloc[0]['ymax']
            
            #Reading the bounding box (OCR)
            #using PyTesseract
            roi_ocr_text = pytesseract.image_to_string(img[int(ymin) : int(ymax), int(xmin) : int(xmax)])
            #if roi_ocr_text != "" and roi_ocr_text[-1] == '\n':
            #    roi_ocr_text = roi_ocr_text[:-1]
            req_values[class_list[i]] = roi_ocr_text.strip("\n| ")

            bounding_boxes.append([xmin, ymin, xmax, ymax, class_list[i]])
    
    #Adding the bounding boxes to the image
    #to store the annotated image
    for i in bounding_boxes:
        cv2.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (36, 255, 12), 2)
        cv2.putText(img, i[4], (int(i[0]), int(i[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(PathConstants.ANNOTATED_IMAGE_PATH + filename.split()[0] + "_annotated.jpg", img)
    print(str(count) + " annotated image(s) saved.")

    #Exporting to JSON
    with open(PathConstants.JSON_FILE_PATH + filename.split()[0] + "_json.json", "w") as outfile:
        json.dump(req_values, outfile, indent = 2)
    print(str(count) + " JSON file(s) saved.")
    
    #Adding to final output EXCEL/CSV
    temp_df = pd.DataFrame([req_values])
    output_excel = output_excel.append(temp_df, ignore_index = True)

    count += 1

#Exporting the final output EXCEL/CSV
output_excel.to_excel(PathConstants.OUTPUT_EXCEL + "output_excel.xlsx", index = False)
output_excel.to_csv(PathConstants.OUTPUT_EXCEL + "output_csv.csv", index = False)

print("Success")

#---------------------------------------End of Code------------------------------------

#Approach 1 - Stale
'''import easyocr

#Generate OCR text for each image
#using Tesseract
for file in os.listdir(PathConstants.DATASET_FILE_PATH):
    image = cv2.imread(PathConstants.DATASET_FILE_PATH + file)
    image_ocr_text = pytesseract.image_to_string(image)
    textfile = open(PathConstants.TESSERACT_IMAGE_TEXT_FILE_PATH + file + ".txt", "w")
    textfile.write(image_ocr_text)
    textfile.close()

#using EasyOCR
reader = easyocr.Reader(['en'])
for file in os.listdir(PathConstants.DATASET_FILE_PATH):
    image = cv2.imread(PathConstants.DATASET_FILE_PATH + file)
    image_ocr_text = reader.readtext(image)
    ocr_lines = ""
    for i in image_ocr_text:
        ocr_lines += i[1] + '\n'
    textfile = open(PathConstants.EASYOCR_IMAGE_TEXT_FILE_PATH + file + ".txt", "w")
    textfile.write(ocr_lines)
    textfile.close()'''
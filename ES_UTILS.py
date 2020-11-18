

import cv2
import requests
import json
import pandas as pd

import numpy as np
import imutils
from pyzbar import pyzbar

import re

def get_sticker_2(image_path, padding_ratio):
    cv2_img = cv2.imread(image_path)
    height, width, channel = cv2_img.shape
    image_content = cv2_img.astype('uint8').tolist()
    
    URL = 'http://localhost:8502/v1/models/sticker:predict'
    headers = {"content-type": "application/json"}
    body = {"instances": [{"inputs": image_content}]}
    r = requests.post(URL, data=json.dumps(body), headers = headers) 

    result = json.loads(r.text)['predictions'][0]

    boxes = result['detection_boxes']
    scores = result['detection_scores']

    df_boxes = pd.DataFrame(boxes,columns=['ymin','xmin','ymax','xmax'])
    df_scores = pd.DataFrame(scores,columns=['scores'])

    df = pd.concat([df_boxes,df_scores],axis=1)

    # artificial padding based on image size
    padding_horizontal = width * padding_ratio
    padding_vertical = height * padding_ratio

    # if df['scores'].iloc[0] < 0.7:
    #     raise Exception('confidence score < 0.7. The confidence score was: {}'.format(scores[0][0]))

    # calculate bbox pixel with paddings
    crop_img = []
    for n in [1,3]:
        bbox_miny = int(max(0,df['ymin'].iloc[0] * height - padding_vertical*n))
        bbox_minx = int(max(0,df['xmin'].iloc[0] * width - padding_horizontal*n))
        bbox_maxy = int(df['ymax'].iloc[0] * height + padding_vertical*n)
        bbox_maxx = int(df['xmax'].iloc[0] * width + padding_horizontal*n)

        crop_img.append(cv2_img[bbox_miny:bbox_maxy, bbox_minx:bbox_maxx])

    return crop_img[0], crop_img[1]

def scan_barcode(cv2_image):
    # import the necessary packages
    from pyzbar import pyzbar

    # load the input image
    image = cv2_image
    # find the barcodes in the image and decode each of the barcodes
    barcodes = pyzbar.decode(image, symbols=[pyzbar.pyzbar.ZBarSymbol.CODE39]) # https://github.com/NaturalHistoryMuseum/pyzbar/issues/29

    # loop over the detected barcodes
    for barcode in barcodes:
        # extract the bounding box location of the barcode and draw the
        # bounding box surrounding the barcode on the image
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # the barcode data is a bytes object so if we want to draw it on
        # our output image we need to convert it to a string first
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        # draw the barcode data and barcode type on the image
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2)
        # print the barcode type and data to the terminal
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))   
    return image

def scan_barcode_2(cv2_image):
    # import the necessary packages
    from pyzbar import pyzbar

    # load the input image
    image = cv2_image
    # find the barcodes in the image and decode each of the barcodes
    barcodes = pyzbar.decode(image)

    # loop over the detected barcodes
    output = []
    for barcode in barcodes:
        # extract the bounding box location of the barcode and draw the
        # bounding box surrounding the barcode on the image
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # the barcode data is a bytes object so if we want to draw it on
        # our output image we need to convert it to a string first
        barcodeData = barcode.data.decode("utf-8")
        # add text to list
        output.append(barcodeData.replace(' ',''))
    return output

import pytesseract
def run_tesseract(cv2_img):
    # Adding custom options
    custom_config = r'--oem 1 --psm 6'
    # Run tesseract
    text = pytesseract.image_to_string(cv2_img, config=custom_config)
    #remove blank lines or lines with only space characters
    text = "\n".join([s for s in text.splitlines() if s.replace(" ","")])
    print(run_regex(text))
    cv2.putText(cv2_img, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2) 
    return cv2_img

def run_tesseract_2(cv2_img):
    # Adding custom options
    custom_config = r'--oem 1 --psm 6'
    # Run tesseract
    text = pytesseract.image_to_string(cv2_img, config=custom_config)
    #remove blank lines or lines with only space characters
    return [s for s in text.splitlines() if s.replace(" ","")]

def rotate_image(cv2_image, angular_bound, angular_step):
    #return [imutils.rotate_bound(cv2_image, angle) for angle in np.arange(-angular_bound, angular_bound, angular_step)]
    return [imutils.rotate_bound(cv2_image, angle) for angle in [i*((-1)**i)//2*angular_step for i in range(0,angular_bound//angular_step*2+1)]]

# Define Regex
regex = {
    'date': '([0123]\d[/-]([0123]\d|\w{3})[/-]\d{4})|(\d{4}[/-]([0123]\d|\w{3})[/-][0123]\d)',
    'date_time': '[0123]\d/\d{2}/\d{4} +\d{2}:\d{2}(?::\d{2})*',
    'HKID': '[A-Z]\d{6}\([\d\w]\)',
    'HKID_exact': '^([A-Z]\d{6}[\dA-Z])$',
}
def process_variant(variant, data):
    if "HKID_from_barcode" not in data.keys() or "BarcodeID" not in data.keys():
        for barcode in scan_barcode_2(variant):
            if re.search(regex['HKID_exact'],barcode):
                data['HKID_from_barcode'] = barcode
            else:
                data['BarcodeID'] = barcode

    if "HKID_from_text" not in data.keys() or "date_time" not in data.keys() or "date" not in data.keys():
        for text in run_tesseract_2(variant):
            # search for HKID
            if re.search(regex['HKID'],text):
                data['HKID_from_text'] = re.search(regex['HKID'],text).group()
            # search date time and date
            if re.search(regex['date_time'],text):
                if "date_time" not in data.keys():
                    data['date_time'] = text
            elif re.search(regex['date'],text):
                new_date = re.search(regex['date'],text).group()
                if "date" not in data.keys():
                    data['date'] = new_date
                elif new_date not in data['date']:
                    data['date'] = data['date'] + ' ' + new_date
                    
    return data




def process_image(cv2_sticker):
    # load the image from disk
    image = cv2_sticker
    import pandas as pd
    data = {
        'BarcodeID': '',
        'HKID_from_barcode': '',
        'HKID_from_text': '',
        'date_time': '',
        'date': '',
    }
    # rotate image with no part of the image is cut off
    for rotated in rotate_image(image, 15, 3):
        for barcode in scan_barcode_2(rotated):
            if re.search(regex['HKID_exact'],barcode):
                data['HKID_from_barcode'] = barcode
            else:
                data['BarcodeID'] = barcode

        for text in run_tesseract_2(rotated):
            # search for HKID
            if re.search(regex['HKID'],text):
                data['HKID_from_text'] = re.search(regex['HKID'],text).group()
            # search date time and date
            if re.search(regex['date_time'],text):
                data['date_time'] = text
            elif re.search(regex['date'],text):
                new_date = re.search(regex['date'],text).group()
                if new_date not in data['date']:
                    data['date'] = data['date'] + ' ' + new_date
    return data
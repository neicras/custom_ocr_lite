

# get file paths
import os
import cv2
import pandas as pd
import ES_UTILS

from multiprocessing import Pool

testDirectory = '/Users/EricSan/Custom_OCR_Lite/images/test'

def process_single_file(path):
    print(path)
    full_path = os.path.join(testDirectory, path)

    # segment patient sticker with custom object detection
    processing_segment_1xPad, processing_segment_3xPad = ES_UTILS.get_sticker_2(
        image_path = full_path, 
        padding_ratio = 0.015
    )

    print(path, "#1")

    # get rotation_variance
    segment_variants_1xPad = ES_UTILS.rotate_image( # (cv2_image, angular_bound, angular_step)
        cv2_image = processing_segment_1xPad, 
        angular_bound = 15, 
        angular_step = 3
    )

    segment_variants_3xPad = ES_UTILS.rotate_image( # (cv2_image, angular_bound, angular_step)
        cv2_image = processing_segment_3xPad, 
        angular_bound = 15, 
        angular_step = 3
    )

    fullecg_variants = ES_UTILS.rotate_image( # (cv2_image, angular_bound, angular_step)
        cv2_image = cv2.imread(full_path), 
        angular_bound = 2, 
        angular_step = 3
    )

    data = {}
    for variant in segment_variants_1xPad:
        data = ES_UTILS.process_variant(variant, data)
    for variant in segment_variants_3xPad:
        data = ES_UTILS.process_variant(variant, data)
    for variant in fullecg_variants:
        data = ES_UTILS.process_variant(variant, data)

    data['ECG_PATH'] = path

    print(data)

    return data

def process_single_file_1(filepath):
    data = {
        'path': filepath,
        'id': 'Y1745709',
        'date': '07/11/2020',
    }
    return data

def main():
    
    output_df = pd.DataFrame()
    
    paths = os.listdir(testDirectory)

    # non multiprocessing    
    for path in paths[:10]: 
        data = process_single_file(path)
        output_df = output_df.append(data, ignore_index=True)

    # # multiprocessing
    # with Pool(2) as p:
    #     output_data_list = p.map(process_single_file, paths[:10])
    # for data in output_data_list:
    #     output_df = output_df.append(data, ignore_index=True)

    output_df.to_csv('output_df.csv')

import time
start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
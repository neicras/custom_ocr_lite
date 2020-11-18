

# get file paths
import os
import cv2
import pandas as pd
import ES_UTILS
import sqlite3 as sql

from multiprocessing import Pool

testDirectory = '/home/medicine/development/decg_custom_ocr_lite/images/test'
targetDirectory = '/home/medicine/development/decg_v2_202005_c/media/ecg'

def process_single_file(path):

    start_time = time.time()

    full_path = os.path.join(targetDirectory, path)

    # initialize data
    data = {}
    data['ECG_PATH'] = path

    # ad-hoc: wrap in a try, except for error
    try:
        # segment patient sticker with custom object detection
        processing_segment_1xPad, processing_segment_3xPad = ES_UTILS.get_sticker_2(
            image_path = full_path, 
            padding_ratio = 0.015
        )
    except:
        data['BarcodeID'] = 'error'
        data['HKID_from_barcode'] = 'error'
        data['HKID_from_text'] = 'error'
        return data

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
        cv2_image = cv2.resize(
            cv2.imread(full_path), 
            (0,0), 
            fx=1.0, 
            fy=1.0
        ), 
        angular_bound = 15, 
        angular_step = 3
    )

    for variant in segment_variants_1xPad:
        data = ES_UTILS.process_variant(variant, data)
    for variant in segment_variants_3xPad:
        data = ES_UTILS.process_variant(variant, data)
    for variant in fullecg_variants:
        data = ES_UTILS.process_variant(variant, data)

    print(path, "completed at %s seconds ---" % (time.time() - start_time))

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
    paths = os.listdir(targetDirectory)

    # # non multiprocessing and write to csv
    # for path in paths[:10]: 
    #     data = process_single_file(path)
    #     output_df = output_df.append(data, ignore_index=True)
    # output_df.to_csv('output_df.csv')

    # # multiprocessing and write to csv
    # with Pool(4) as p:
    #     output_data_list = p.map(process_single_file, paths[:20])
    # for data in output_data_list:
    #     output_df = output_df.append(data, ignore_index=True)
    # output_df.to_csv('output_df.csv')

    # # multiprocessing and write to .db
    done_filepaths = []
    conn = sql.connect('output.db')
    done_filepaths = []
    def done_filepaths():
        try:
            df = pd.read_sql('SELECT * FROM output20201108', conn)
            done_filepaths = df['ECG_PATH'].tolist()
            return done_filepaths
        except:
            return done_filepaths

    while set(paths) > set(done_filepaths()):
        output_df = pd.DataFrame()
        with Pool(4) as p:
            output_data_list = p.map(process_single_file, list(set(paths)-set(done_filepaths()))[:100])
        for data in output_data_list:
            output_df = output_df.append(data, ignore_index=True)
        output_df.to_sql('output20201108', conn, if_exists = 'append')

        print("COMPLETED BATCH UPDATED TO OUTPUT.DB")
        print("--- %s files completed ---" % len(set(done_filepaths())))
        print("--- %s seconds ---" % (time.time() - start_time))



import time
start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
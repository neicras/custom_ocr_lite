

# get file paths
import os
import cv2
import pandas as pd
import ES_UTILS

testDirectory = '/Users/EricSan/Custom_OCR_Lite/images/validation'

def main():
    
    output_df = pd.DataFrame()
    
    paths = os.listdir(testDirectory)
    for path in paths[:1]:
        full_path = os.path.join(testDirectory, path)
        
        # segment patient sticker with custom object detection
        processing_segment_1xPad, processing_segment_3xPad = ES_UTILS.get_sticker_2(
            image_path = full_path, 
            padding_ratio = 0.015
        )

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
        output_df = output_df.append(data, ignore_index=True)

    output_df.to_csv('output_df.csv')

import time
start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
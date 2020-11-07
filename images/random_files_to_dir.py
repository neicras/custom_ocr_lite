

import shutil, random, os
dirpath = '/Volumes/QMEcgFtp/home/medicine/development/decg_v2_202005_c/media/ecg'
valDirectory = '/Users/EricSan/Custom_OCR_Lite/images/validation'
val_size = 100

filenames = random.sample(os.listdir(dirpath), val_size)

# copy over testset
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    shutil.copyfile(srcpath, os.path.join(valDirectory,fname))






import shutil, random, os
dirpath = '/home/medicine/development/decg_v2_202005_c/media/ecg'
valDirectory = '/home/medicine/development/decg_custom_ocr_lite/images/test'
val_size = 20

filenames = random.sample(os.listdir(dirpath), val_size)

# copy over testset
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    shutil.copyfile(srcpath, os.path.join(valDirectory,fname))




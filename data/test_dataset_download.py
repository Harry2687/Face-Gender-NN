import gdown
import zipfile
import os
import shutil

url = 'https://drive.google.com/file/d/1_oB7QX2rn8-kRTI9uk0KtGIn4DijYDE0/view?usp=sharing'
output = 'download.zip'

gdown.download(url, output, fuzzy=True)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('data/')

os.remove(output)
shutil.rmtree('data/__MACOSX')
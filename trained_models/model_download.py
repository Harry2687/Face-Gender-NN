import gdown

url = 'https://drive.google.com/file/d/1_mYn2LrhG080Xvt26tWBtJ8U_0F2E1-s/view?usp=sharing'
output = 'trained_models/resnetModel_128_epoch_2.pt'

gdown.download(url, output, fuzzy=True)
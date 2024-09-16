import os
import PIL
from PIL import Image
import cv2
import numpy as np

target_size = (3584, 3584)

list = os.listdir('./data/raw')
os.makedirs('./data/resized')
print(len(list))

for folder in list:
    os.makedirs("./data/resized/"+folder)
    for subfolder in os.listdir("./data/raw/"+folder):
        i = 1
        os.makedirs("./data/resized/"+folder+"/"+subfolder)
        for item in os.listdir("./data/raw/"+folder+"/"+subfolder):
            print('Converting...{0}'.format(i))
            name = os.path.basename(item)
            image = np.array(Image.open(os.path.join('./data/raw/'+folder+"/"+subfolder,name)))
            name = os.path.splitext(name)
            resized = cv2.resize(image, target_size)
            cv2.imwrite(os.path.join('./data/resized/'+folder+"/"+subfolder,name[0]+'.png'), resized)
            i += 1

print('Finished convert')


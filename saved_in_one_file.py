import os
import random

import cv2

def saved_in_one_file():
    count = 0
    if os.getcwd() != '/home/fish/图片/shells':
        os.chdir('/home/fish/图片/shells')
    dir = '/home/fish/图片/all_in_one'
    if not os.path.exists(dir):
        os.mkdir(dir)
    for root, dirs, files in os.walk(os.getcwd()):
        if dirs == []:
            print(root)
            c = input('wait')
            for file in files:
                if file.split('.')[-1] == 'txt':
                    continue
                img = cv2.imread(os.path.join(root, str(file)))
                if img.shape == None:
                    print('error')
                    continue
                    
                if img.shape != (256, 256, 3):
                    raise ValueError("Image size %d doesn't match 256,256,3 " % img.shape)
                
                cv2.imwrite(os.path.join(dir, str(file)), img)

def main():
    saved_in_one_file()

if __name__ == '__main__':
    main()
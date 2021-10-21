import json
from skimage.measure import compare_ssim
import cv2

import imageio

import multiprocessing

def read_lines(file):
    arr = []
    with open(file, 'r') as f:
        arr = f.readlines()
    return arr




from multiprocessing.dummy import Pool as ThreadPool
global root, dirs, files,tmp_num
img_diff = dict()
count = 0





def multiplication(video):
    video_name = (video.split('/')[4]).split('.')[0]

    img = list()
    global count
    global img_diff
    img_diff[video_name] = list()
    try:

        vid = imageio.get_reader(video.split(' ')[0], 'ffmpeg')
        for num, im in enumerate(vid):
            img.append(im)
        for i in range(len(img) - 1):
            tmp1 = cv2.cvtColor(img[i], cv2.COLOR_RGB2GRAY)
            tmp2 = cv2.cvtColor(img[i + 1], cv2.COLOR_RGB2GRAY)
            (score, diff) = compare_ssim(tmp1, tmp2, full=True)
            score = 1 - score

            img_diff[video_name].append(score)
        count = count + 1
        print(count)
    except(OSError):
        print('error!')
        print(video_name)




video_list = read_lines('/home/zhiyuan/mmaction2/train1.csv')[:]

pool = ThreadPool(processes=28)
re = pool.map(multiplication, video_list)
fileObject = open('img_diff_kinetics_train_1.json', 'a+')
jsonData = json.dumps(img_diff)
fileObject.write(jsonData)
fileObject.close()
pool.close()
pool.join()
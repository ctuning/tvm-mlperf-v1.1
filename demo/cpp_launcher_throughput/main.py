from async_launcher import AsyncGraphExecutor
import numpy as np
from PIL import Image

import sys


def top5(arr):
    N = arr.shape[0]
    topN_res = []
    for i in range(N):
        arr_ = np.squeeze(arr[i, :])
        n = 5
        idxs = (-arr_).argsort()[:n]
        res = [(i, arr_[i]) for i in idxs]
        res.sort(key=lambda a: a[1], reverse=True)
        topN_res.append(res)
    return topN_res


def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i, :, :] = (img_data[i, :, :]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


def get_img(img_path, batch=1, shape=[224, 224]):
    img = Image.open(img_path)
    img = img.resize(size=shape)
    in_data = np.asarray(img)
    in_data = in_data.transpose((2, 0, 1))
    shape = in_data.shape
    in_data = preprocess(in_data)
    in_data = np.broadcast_to(in_data.astype("float32"), shape=(batch, *shape))
    return in_data


def main(mod_path, img_path):
    print("Hello")
    executor = AsyncGraphExecutor(mod_path)
    executor.initialize_for_thread()

    # input_img = np.zeros([3, 224, 224], dtype=np.float32)
    input_img = get_img(img_path, batch=3, shape=[224, 224])

    res = executor.infer([input_img])

    print("CAT is expected")
    print(top5(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

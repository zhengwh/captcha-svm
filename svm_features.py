"""
将图片的像素信息，利用特征工程，降维 为 特征列表文件
"""
import os
from os.path import join

from PIL import Image

from cfg import train_file_name, cut_pic_folder, test_feature_file, test_cut_pic_folder


def get_feature(img):
    """
    获取指定图片的特征值,
    1. 按照每排的像素点,高度为10,然后宽度为6,总共16个维度
    2. 计算每个维度（行 或者 列）上有效像素点的和

    :type img: Image
    :return:一个维度为16的列表
    """

    width, height = img.size

    pixel_cnt_list = []
    height = 10
    for y in range(height):
        pix_cnt_x = 0
        for x in range(width):
            if img.getpixel((x, y)) == 0:  # 黑色点
                pix_cnt_x += 1

        pixel_cnt_list.append(pix_cnt_x)

    for x in range(width):
        pix_cnt_y = 0
        for y in range(height):
            if img.getpixel((x, y)) == 0:  # 黑色点
                pix_cnt_y += 1

        pixel_cnt_list.append(pix_cnt_y)

    return pixel_cnt_list


def get_svm_train_txt():
    """
    获取 测试集 的像素特征文件。
    所有的数字的可能分类为10，分别放在以相应的数字命名的目录中
    :return:
    """
    svm_feature_file = open(train_file_name, 'w')

    for i in range(10):
        img_folder = join(cut_pic_folder, str(i))
        convert_imgs_to_feature_file(i, svm_feature_file, img_folder)

    # 不断地以追加的方式写入到同一个文件当中
    svm_feature_file.close()


def get_svm_test_txt():
    """
    获取 测试集 的像素特征文件
    :return:
    """

    img_folder = test_cut_pic_folder
    test_file = open(test_feature_file, 'w')
    convert_imgs_to_feature_file(8, test_file, img_folder)  # todo 先用0代替
    test_file.close()


def convert_imgs_to_feature_file(dig, svm_feature_file, img_folder):
    """
    将某个目录下二进制图片文件，转换成特征文件
    :param dig:检查的数字
    :param svm_feature_file: svm的特征文件完整路径
    :type dig:int
    :return:
    """
    file_list = os.listdir(img_folder)

    # sample_cnt = 0
    # right_cnt = 0
    for file in file_list:
        img = Image.open(img_folder + '/' + file)
        dif_list = get_feature(img)
        # sample_cnt += 1
        line = convert_values_to_str(dig, dif_list)
        svm_feature_file.write(line)
        svm_feature_file.write('\n')


def convert_values_to_str(dig, dif_list):
    """
    将特征值串转化为标准的svm输入向量:

    9 1:4 2:2 3:2 4:2 5:3 6:4 7:1 8:1 9:1 10:3 11:5 12:3 13:3 14:3 15:3 16:6

    最前面的是 标记值，后续是特征值
    :param dif_list:
    :type dif_list: list[int]
    :return:
    """
    index = 1
    line = '%d' % dig

    for item in dif_list:
        fmt = ' %d:%d' % (index, item)
        line += fmt
        index += 1

    # print(line)
    return line


def convert_feature_to_vector(feature_list):
    """

    :param feature_list:
    :return:
    """
    index = 1
    xt_vector = []
    feature_dict = {}
    for item in feature_list:
        feature_dict[index] = item
        index += 1
    xt_vector.append(feature_dict)
    return xt_vector


if __name__ == '__main__':
    print("start captcha app...")

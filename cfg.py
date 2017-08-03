"""
配置的config文件
"""

from os.path import join

# 所有图片的训练目录
home_root = '/home/harmo'  # 用户的home目录
data_root = join(home_root, 'work/crack/my-capt-data/capt-simple')  # todo 上传之前，必须删除此目录

origin_pic_folder = join(data_root, 'origin')  # 原始图像目录
bin_clear_folder = join(data_root, 'bin_clear')  # "原始图像 -> 二值 -> 除噪声" 之后的图片文件目录
cut_pic_folder = join(data_root, 'cut_pic')  # 1张4位验证字符图片 -> 4张单字符图片。然后再将相应图片拖动到指定目录，完全数据标记工作

test_cut_pic_folder = join(data_root, 'cut_test')  # 一组全为 8 的图片集，用于做简单的模型验证测试

img_path = data_root + '/train_origin/svm_ocr-simple-char-captcha-origin.bmp'

# SVM训练相关路径
svm_root = join(data_root, 'svm_train')  # 用于SVM训练的特征文件
train_file_name = join(svm_root, 'train_pix_feature_xy.txt')  # 保存训练集的 像素特征文件
test_feature_file = join(svm_root, 'last_test_pix_xy_8.txt')  # 只以一组8数字的特征文件为例子来做简单的验证测试
model_path = join(svm_root, 'svm_model_file')  # 训练完毕后，保存的SVM模型参数文件

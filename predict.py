"""
在训练结束后，一个完整的hack流程：

1. 从网站动态请求相应的验证文件
#. 进行图像预处理
#. 将图像进行分割成最小基本单位
#. 计算出本图像的特征
#. 使用SVM训练好的模型进行

对新的验证图片来做结果预测

"""
import io

import requests
from PIL import Image
from dtlib.randtool import get_uuid1_key

from cfg import data_root
from img_tools import get_clear_bin_image, get_crop_imgs
from lib.svmutil import svm_predict, svm_load_model
from svm_features import get_feature, convert_feature_to_vector


def crack_captcha():
    """
    破解验证码,完整的演示流程
    :return:
    """

    # 向指定的url请求验证码图片
    rand_captcha_url = 'http://www.captcha.com/randcode/take-your-own-url'
    res = requests.get(rand_captcha_url, stream=True)

    f = io.BytesIO()
    for chunk in res.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            f.write(chunk)
            f.flush()

    img = Image.open(f)  # 从网络上请求验证码图片保存在内存中
    bin_clear_img = get_clear_bin_image(img)  # 处理获得去噪的二值图
    child_img_list = get_crop_imgs(bin_clear_img)  # 切割图片为单个字符，保存在内存中,例如：4位验证码就可以分割成4个child

    # 加载SVM模型进行预测
    svm_model_name = 'svm_model_file'
    model_path = data_root + '/svm_train/' + svm_model_name
    model = svm_load_model(model_path)

    img_ocr_name = '__'
    for child_img in child_img_list:
        img_feature_list = get_feature(child_img)  # 使用特征算法，将图像进行特征化降维

        yt = [0]  # 测试数据标签
        # xt = [{1: 1, 2: 1}]  # 测试数据输入向量
        xt = convert_feature_to_vector(img_feature_list)  # 将所有的特征转化为标准化的SVM单行的特征向量
        p_label, p_acc, p_val = svm_predict(yt, xt, model)
        img_ocr_name += ('%d' % p_label[0])  # 将识别结果合并起来

    uuid_tag = get_uuid1_key()  # 生成一组随机的uuid的字符串（开发人员自己写，比较好实现）

    img_save_folder = data_root + '/crack_img_res'
    img.save(img_save_folder + '/' + img_ocr_name + '__' + uuid_tag + '.png')
    # 例如：__0067__77b10a28f73311e68abef0def1a6bbc8.png
    f.close()


def crack_100():
    """
    直接从在线网上下载100张图片，然后识别出来
    :return:
    """
    for i in range(200):
        crack_captcha()


if __name__ == '__main__':
    crack_100()
    pass

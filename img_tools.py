"""
一些图像处理工具
"""
import os
from PIL import Image
from cfg import img_path, bin_clear_folder, origin_pic_folder, cut_pic_folder, data_root
from os.path import join


def get_bin_table(threshold=140):
    """
    获取灰度转二值的映射table
    :param threshold:
    :return:
    """
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    return table


def sum_9_region(img, x, y):
    """
    9邻域框,以当前点为中心的田字框,黑点个数,作为移除一些孤立的点的判断依据
    :param img: Image
    :param x:
    :param y:
    :return:
    """
    cur_pixel = img.getpixel((x, y))  # 当前像素点的值
    width = img.width
    height = img.height

    if cur_pixel == 1:  # 如果当前点为白色区域,则不统计邻域值
        return 0

    if y == 0:  # 第一行
        if x == 0:  # 左上顶点,4邻域
            # 中心点旁边3个点
            sum = cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y + 1))
            return 4 - sum
        elif x == width - 1:  # 右上顶点
            sum = cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y + 1))

            return 4 - sum
        else:  # 最上非顶点,6邻域
            sum = img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y + 1)) \
                  + cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y + 1))
            return 6 - sum
    elif y == height - 1:  # 最下面一行
        if x == 0:  # 左下顶点
            # 中心点旁边3个点
            sum = cur_pixel \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y - 1)) \
                  + img.getpixel((x, y - 1))
            return 4 - sum
        elif x == width - 1:  # 右下顶点
            sum = cur_pixel \
                  + img.getpixel((x, y - 1)) \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y - 1))

            return 4 - sum
        else:  # 最下非顶点,6邻域
            sum = cur_pixel \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x, y - 1)) \
                  + img.getpixel((x - 1, y - 1)) \
                  + img.getpixel((x + 1, y - 1))
            return 6 - sum
    else:  # y不在边界
        if x == 0:  # 左边非顶点
            sum = img.getpixel((x, y - 1)) \
                  + cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x + 1, y - 1)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y + 1))

            return 6 - sum
        elif x == width - 1:  # 右边非顶点
            # print('%s,%s' % (x, y))
            sum = img.getpixel((x, y - 1)) \
                  + cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x - 1, y - 1)) \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y + 1))

            return 6 - sum
        else:  # 具备9领域条件的
            sum = img.getpixel((x - 1, y - 1)) \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y + 1)) \
                  + img.getpixel((x, y - 1)) \
                  + cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x + 1, y - 1)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y + 1))
            return 9 - sum


def remove_noise_pixel(img, noise_point_list):
    """
    根据噪点的位置信息，消除二值图片的黑点噪声
    :type img:Image
    :param img:
    :param noise_point_list:
    :return:
    """
    for item in noise_point_list:
        img.putpixel((item[0], item[1]), 1)


def get_clear_bin_image(image):
    """
    获取干净的二值化的图片。
    图像的预处理：
    1. 先转化为灰度
    2. 再二值化
    3. 然后清除噪点
    参考:http://python.jobbole.com/84625/
    :type img:Image
    :return:
    """
    imgry = image.convert('L')  # 转化为灰度图

    table = get_bin_table()
    out = imgry.point(table, '1')  # 变成二值图片:0表示黑色,1表示白色

    noise_point_list = []  # 通过算法找出噪声点,第一步比较严格,可能会有些误删除的噪点
    for x in range(out.width):
        for y in range(out.height):
            res_9 = sum_9_region(out, x, y)
            if (0 < res_9 < 3) and out.getpixel((x, y)) == 0:  # 找到孤立点
                pos = (x, y)  #
                noise_point_list.append(pos)
    remove_noise_pixel(out, noise_point_list)
    return out


def get_crop_imgs(img):
    """
    按照图片的特点,进行切割,这个要根据具体的验证码来进行工作. # 见本例验证图的结构原理图
    分割图片是传统机器学习来识别验证码的重难点，如果这一步顺利的话，则多位验证码的问题可以转化为1位验证字符的识别问题
    :param img:
    :return:
    """
    child_img_list = []
    for i in range(4):
        x = 2 + i * (6 + 4)  # 见原理图
        y = 0
        child_img = img.crop((x, y, x + 6, y + 10))
        child_img_list.append(child_img)

    return child_img_list


def print_line_x(img, x):
    """
    打印一个Image图像的第x行，方便调试
    :param img:
    :type img:Image
    :param x:
    :return:
    """
    print("line:%s" % x)
    for w in range(img.width):
        print(img.getpixel((w, x)), end='')
    print('')


def print_bin(img):
    """
    输出二值后的图片到控制台，方便调试的函数
    :param img:
    :type img: Image
    :return:
    """
    print('current binary output,width:%s-height:%s\n')
    for h in range(img.height):
        for w in range(img.width):
            print(img.getpixel((w, h)), end='')
        print('')


def save_crop_imgs(bin_clear_image_path, child_img_list):
    """
    输入：整个干净的二化图片
    输出：每张切成4版后的图片集
    保存切割的图片

    例如： A.png ---> A-1.png,A-2.png,... A-4.png 并保存，这个保存后需要去做label标记的
    :param bin_clear_image_path: xxxx/xxxxx/xxxxx.png 主要是用来提取切割的子图保存的文件名称
    :param child_img_list:
    :return:
    """
    full_file_name = os.path.basename(bin_clear_image_path)  # 文件名称
    full_file_name_split = full_file_name.split('.')
    file_name = full_file_name_split[0]
    # file_ext = full_file_name_split[1]

    i = 0
    for child_img in child_img_list:
        cut_img_file_name = file_name + '-' + ("%s.png" % i)
        child_img.save(join(cut_pic_folder, cut_img_file_name))
        i += 1


# 训练素材准备：文件目录下面的图片的批量操作

def batch_get_all_bin_clear():
    """
    训练素材准备。
    批量操作：获取所有去噪声的二值图片
    :return:
    """

    file_list = os.listdir(origin_pic_folder)
    for file_name in file_list:
        file_full_path = os.path.join(origin_pic_folder, file_name)
        image = Image.open(file_full_path)
        get_clear_bin_image(image)


def batch_cut_images():
    """
    训练素材准备。
    批量操作：分割切除所有 "二值 -> 除噪声" 之后的图片，变成所有的单字符的图片。然后保存到相应的目录，方便打标签
    """

    file_list = os.listdir(bin_clear_folder)
    for file_name in file_list:
        bin_clear_img_path = os.path.join(bin_clear_folder, file_name)
        img = Image.open(bin_clear_img_path)

        child_img_list = get_crop_imgs(img)
        save_crop_imgs(bin_clear_img_path, child_img_list)  # 将切割的图进行保存，后面打标签时要用


# 中间的demo效果演示


def demo_cut_pic():
    """
    做实验研究时的演示代码
    :return:
    """
    img_path = join(data_root, 'demo-6937/ocr-simple-char-captcha-bin-clear-6937.png')
    img = Image.open(img_path)
    cut_save = data_root + '/demo-6937'
    child_img_list = get_crop_imgs(img)

    index = 0
    for child_img in child_img_list:
        child_img.save(cut_save + '/cut-%d.png' % index)
        index += 1


def get_bin_img_name(img_path):
    """
    根据原始origin 文件路径,获取二值而且去噪声的文件路径
    :param img_path:
    :type img_path:str
    :return:
    """
    path_split = img_path.split('/')
    file_name_split = path_split[-1].split('.')
    file_name = file_name_split[0]  # 文件名
    # file_ext = file_name_split[1]  # 扩展名

    new_file = '/'.join(item for item in path_split[:-2]) + '/bin_clear/' + file_name + '.png'
    return new_file


def demo_handle_save_bin_clear_pic(image):
    """
    图像处理函数的演示
    在训练分析阶段的时候使用:保存二次的二值图,
    :type img:Image
    :return:
    """
    out = get_clear_bin_image(image)
    new_file_path = get_bin_img_name(img_path)
    print(new_file_path)
    out.save(new_file_path)


if __name__ == "__main__":
    print(get_bin_table())
    # batch_get_all_bin_clear()  # 获取所有的二值化的初步去噪的图片
    # cut_all_pic()  # 切割图片成单个字
    # save_train_txt()
    # save_test_txt()
    # crack_captcha()
    # img = Image.open(img_path)
    # handle_save_bin_clear_pic(img)
    # demo_cut_pic()
    pass

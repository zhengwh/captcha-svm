"""
SVM示例:

- http://www.cnblogs.com/Finley/p/5329417.html
"""
from cfg import model_path, train_file_name, test_feature_file
from lib.svm import svm_problem, svm_parameter
from lib.svmutil import *


def svm_data_demo():
    """
    这个是来自于网上的demo，和本识图项目无关
    :return:
    """
    y = [1, -1]  # 训练数据的标签
    x = [{1: 1, 2: 1}, {1: -1, 2: -1}]  # 训练数据的输入向量
    # <label> <index1>:<value1> <index2>:<value2>
    # 相当于找到的特征值

    prob = svm_problem(y, x)  # 定义SVM模型的训练数据
    param = svm_parameter('-t 0 -c 4 -b 1')  # 训练SVM模型所需的各种参数
    model = svm_model_train(prob, param)  # 训练好的SVM模型

    # svm_save_model('model_file', model)#将训练好的模型保存到文件中

    # 使用测试数据集对已经训练好的模型进行测试
    yt = [-1]  # 测试数据标签
    xt = [{1: 1, 2: 1}]  # 测试数据输入向量

    p_label, p_acc, p_val = svm_predict(yt, xt, model)

    """
    - p_label:预测标签的列表
    - p_acc 存储预测的精确度,均值和回归的平方相关系数
    - p_vals 在指定参数‘-b 1’时将返回的判定系数（判定的可靠程度）
    """

    print(p_label)


def svm_model_train():
    """
    使用图像的特征文件 来训练生成model文件
    :return:
    """

    y, x = svm_read_problem(train_file_name)
    model = svm_train(y, x)
    svm_save_model(model_path, model)


def svm_model_test():
    """
    使用测试集测试模型
    :return:
    """
    yt, xt = svm_read_problem(test_feature_file)
    model = svm_load_model(model_path)
    p_label, p_acc, p_val = svm_predict(yt, xt, model)

    cnt = 0
    for item in p_label:
        print('%d' % item, end=',')

        cnt += 1
        if cnt % 8 == 0:
            print('')


if __name__ == "__main__":
    print('svm demo')
    # train_svm_model()
    svm_model_test()

"""
将result_list分割成训练集和测试集，分割比例是80%，20%。
分割操作使用的是scikit-learn的train_test_split函数，该函数会在分割时随机打乱数据顺序，确保数据的随机性。
"""

from os import path
from glob import glob
import argparse
from sklearn.model_selection import train_test_split
def main(args):

    # 获取当前的目录 LRS2示例:5535415699068794046/00001
    #result = [path.basename(f) for f in glob("{}/*".format(args.base_path)) if path.isfile(f)]
    result=list(path.basename(f) for f in glob("{}/*".format(args.base_path)))
    train_files, test_files = train_test_split(result, test_size=0.2, random_state=42)
    # 将数组写进文本
    #输入参数 --base_path lrw_preprocessed/val
    with open(path.join("filelists",'train_{}.txt'.format(path.basename(args.base_path))), 'w',
              encoding='utf-8') as train_file, \
         open(path.join("filelists",'test_{}.txt'.format(path.basename(args.base_path))), 'w',
              encoding='utf-8') as test_file:
        train_file.write("\n".join(train_files))
        test_file.write("\n".join(test_files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', help='get the sub filelist', default=True, type=str)
    args = parser.parse_args()
    main(args)
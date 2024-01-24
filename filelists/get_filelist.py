
from os import path
from glob import glob
import argparse
def main(args):

    # 获取当前的目录 LRS2示例:5535415699068794046/00001
    #result = [path.basename(f) for f in glob("{}/*".format(args.base_path)) if path.isfile(f)]
    result=list(path.basename(f) for f in glob("{}/*".format(args.base_path)))
    #print(result)
    # 将数组写进文本
    #输入参数 --base_path lrw_preprocessed/val
    with open(path.join("", '{}.txt'.format(path.basename(args.base_path))), 'w', encoding='utf-8') as fi:
        fi.write("\n".join(result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', help='get the sub filelist', default=True, type=str)
    args = parser.parse_args()
    main(args)
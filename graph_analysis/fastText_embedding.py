# https://pypi.org/project/fasttext/
# https://github.com/facebookresearch/fastText
from glove_embedding import load_object
import io
import fasttext
import fasttext.util
import sys
import os
sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# can't n'gram
# data = load_vectors("./data/crawl-300d-2M-subword.vec")
# print(data["newyork"])
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


# can n'gram
def load_model(fname):
    ft = fasttext.load_model(fname)
    print(ft.get_dimension())
    print(ft.get_word_vector('hello').shape)
    print(ft["hello"])
    return ft


if __name__ == '__main__':
    model = load_model("./data/crawl-300d-2M-subword.bin")
    import pdb; pdb.set_trace()
    print()

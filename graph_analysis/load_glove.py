import numpy as np

def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File, 'r', encoding='utf-8')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel), " words loaded!")
    return gloveModel

# load glove
gloveModel = loadGloveModel("./data/glove.42B.300d.txt")
# need find object 300d
objects = open("./data/objects.txt").readlines()
objects = [o.strip() for o in objects]
f = {}
cantf = {}
for o in objects:
    try:
        print(gloveModel[o.lower()])
        f[o.lower()] = gloveModel[o.lower()]
    except Exception as e:
        cantf[o.lower()] = None

# save h5 json
import h5py
hf = h5py.File('./data/glove_300d_{}.h5'.format(len(list(f.values()))), 'w')
for k, v in f.items():
    hf.create_dataset(k, data=v)
hf.close()
# save cantf json
import json
with open("./data/cantf.json", "w") as outfile:
    json.dump(cantf, outfile)
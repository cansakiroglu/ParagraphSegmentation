import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from time import time
import os


#%%
nltk.download('punkt')
nltk.download("stopwords")

#%%
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#%%
all_text = ""
trainRootPath = "/home/mehmet/CS/NLP/wiki_727/train"
devRootPath = "/home/mehmet/CS/NLP/wiki_727/dev"
testRootPath = "/home/mehmet/CS/NLP/wiki_727/test"

# for folder1 in os.listdir(trainRootPath):
#     for folder2 in os.listdir(os.path.join(trainRootPath, folder1)):
#         for folder3 in os.listdir(os.path.join(trainRootPath, folder1, folder2)):
#             with open(os.path.join(trainRootPath, folder1, folder2, folder3), "r") as f:
#                 allText = allText + "".join(f.readlines())
#                 print("Doc {} {} {} is completed",folder1, folder2, folder3)
#
#%%
sub_dirs_train = [x[0] for x in os.walk(trainRootPath)]
#sub_dirs_dev =[x[0] for x in os.walk(devRootPath)]
#sub_dirs_test =[x[0] for x in os.walk(testRootPath)]

#%%
def get_all_text_in_folder(subdirs):
    all_text = ""
    for folder in subdirs:
        print("folder {}", folder)
        for file in os.listdir(folder):
            current = os.path.join(folder,file)
            if os.path.isdir(current):
                continue
            with open(current, "r") as f:
                all_text = all_text + "".join(f.readlines())
        print("Folder {} is completed", folder)

    return all_text.lower()

#%%


all_text_train = get_all_text_in_folder(sub_dirs_train)
#all_text_dev = get_all_text_in_folder(sub_dirs_dev)
#all_text_test = get_all_text_in_folder(sub_dirs_test)


#%%
text_tokenized = sent_tokenize(all_text_train)

#%%
tagged_data_small = [TaggedDocument(word_tokenize(d), [i]) for i, d in enumerate(text_tokenized[:100000])]

#%%
model_small = Doc2Vec(tagged_data_small, vector_size=64, window=10, min_count=2, epochs=40 ,workers=11)


#%%
vec1 = model_small.infer_vector(["Gengis", "khan","defended","his", "country", "succesfully" ])
vec2 = model_small.infer_vector(["He","showed","great", "heroic","efforts","in","this","wars"])
vec3 = model_small.infer_vector(["Gengis", "Khan","has", "12", "kids"])
vec4 = model_small.infer_vector(["Gengis", "khans", "first", "boy", "born", "in", "1220"])


#%%
import numpy as np
from numpy.linalg import  norm
def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

sim1 = cos_similarity(vec1, vec2)
sim2 = cos_similarity(vec2, vec3)
sim3 = cos_similarity(vec1, vec3)
sim4 = cos_similarity(vec3,vec4)

#%%
saveFileName = "/home/mehmet/CS/NLP/wiki727k_embeddings"
model_small.save(saveFileName)

#%%
import re

all_text_train = re.sub("====(=)+", "<p>", all_text_train)
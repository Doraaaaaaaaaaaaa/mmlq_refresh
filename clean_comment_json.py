import io
import os
from tqdm import tqdm
import pickle

comment_file = "AVA_Comments_Full.txt"
ava_path = "../datasets/AVA/ava_captions"

ava_comments = {}

with io.open(os.path.join(ava_path, comment_file), 'r', encoding = 'utf-8') as f:
    for count, line in tqdm(enumerate(f)):
        # print(count)
        elements = line.strip('\n').split('#')
        img={}
        image_path = elements[1]
        captions = elements[2:]

        try:
            # ava_comments[int(image_path)] = " ".join(captions)
            ava_comments[int(image_path)] = captions
        except ValueError:
            continue

with open(os.path.join(ava_path, "AVA_Comments_Full.pkl"), "wb") as f:
    pickle.dump(ava_comments, f, -1)

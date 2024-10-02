import pickle as pkl
import numpy as np
import torch

f = open('/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/human_tracking_detection/YOLOv7-DeepSORT-Human-Tracking/train_set_labels.pkl', 'rb')
annotations_doc = pkl.load(f)
f.close()

labels = list(annotations_doc.values())
#print(labels)
counter = np.zeros(80)
for l in labels:
    for m in l:
        counter[m-1] += 1

print(counter)
np.save("label_num.npy", counter)
import pickle as pkl
import numpy as np
import torchmetrics
import torch




f = open('/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/human_tracking_detection/YOLOv7-DeepSORT-Human-Tracking/train_set_bbox_tracking.pkl', 'rb')
tracking_bboxes = pkl.load(f)
f.close()
f = open('/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/human_tracking_detection/YOLOv7-DeepSORT-Human-Tracking/train_set_keyframe.pkl', 'rb')
key_bboxes = pkl.load(f)
f.close()
f = open('/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/human_tracking_detection/YOLOv7-DeepSORT-Human-Tracking/train_set_labels.pkl', 'rb')
annotations_doc = pkl.load(f)
f.close()



tracking_bboxes = list(tracking_bboxes.values())
annotations = list(annotations_doc.values())
captions = list(captions.values())















f = open('/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/human_tracking_detection/YOLOv7-DeepSORT-Human-Tracking/train_set_description.pkl','rb')
captions = pkl.load(f)
f.close()
f = open('/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/human_tracking_detection/YOLOv7-DeepSORT-Human-Tracking/test_set_bbox_tracking.pkl', 'rb')
tracking_bboxes = pkl.load(f)
f.close()
f = open('/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/human_tracking_detection/YOLOv7-DeepSORT-Human-Tracking/test_set_keyframe.pkl', 'rb')
key_bboxes = pkl.load(f)
f.close()
f = open('/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/human_tracking_detection/YOLOv7-DeepSORT-Human-Tracking/test_set_labels.pkl', 'rb')
annotations_doc = pkl.load(f)
f.close()
f = open('/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/human_tracking_detection/YOLOv7-DeepSORT-Human-Tracking/test_set_description.pkl','rb')
captions = pkl.load(f)
f.close()      
f = open('/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/human_tracking_detection/YOLOv7-DeepSORT-Human-Tracking/val_set_bbox_tracking.pkl', 'rb')
tracking_bboxes = pkl.load(f)
f.close()
f = open('/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/human_tracking_detection/YOLOv7-DeepSORT-Human-Tracking/val_set_keyframe.pkl', 'rb')
key_bboxes = pkl.load(f)
f.close()
f = open('/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/human_tracking_detection/YOLOv7-DeepSORT-Human-Tracking/val_set_labels.pkl', 'rb')
annotations_doc = pkl.load(f)
f.close()
f = open('/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/human_tracking_detection/YOLOv7-DeepSORT-Human-Tracking/val_set_description.pkl','rb')
captions = pkl.load(f)
f.close() 
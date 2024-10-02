"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from typing import Iterable
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
import pickle as pkl
from sklearn.preprocessing import MultiLabelBinarizer
import argparse
import numpy as np
import cv2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from skimage.color import hsv2rgb,rgb2hsv
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from PIL import Image
import torch
"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
from typing import Dict
from omegaconf import OmegaConf

def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


    #Rain
def generate_random_lines(imshape, slant, drop_length, rain_type):
    drops = []
    area = imshape[0] * imshape[1]
    no_of_drops = area // 600

    # if rain_type.lower()=='drizzle':

    if rain_type == 1:
        no_of_drops = area // 770
        drop_length = 10
        # print("drizzle")
    # elif rain_type.lower()=='heavy':
    elif rain_type == 2:
        no_of_drops = area // 770
        drop_length = 30
    # elif rain_type.lower()=='torrential':
    elif rain_type == 3:
        no_of_drops = area // 770
        drop_length = 60
        # print("heavy")
    elif rain_type == 4:
        no_of_drops = area // 500
        drop_length = 60
    elif rain_type == 5:
        no_of_drops = area // 400
        drop_length = 80
        # print('torrential')

    for i in range(no_of_drops):  ## If You want heavy rain, try increasing this
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x, y))
    return drops, drop_length

def rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops, darken):
    imshape = image.shape
    rain_mask = np.zeros((imshape[0], imshape[1]))
    image_t = image.copy()
    for rain_drop in rain_drops:
        cv2.line(rain_mask, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length),
                 drop_color, drop_width)

    rain_mask = np.stack((rain_mask, rain_mask, rain_mask), axis=2)
    image_rain = image + np.array(rain_mask * (1 - image / 255.0) * (1 - np.mean(image) / 255.0), dtype=np.uint8)
    blur_rain = cv2.blur(image_rain, (3, 3))  ## rainy view are blurry
    image_RGB = np.array(blur_rain * rain_mask / 255.0 + image * (1 - rain_mask / 255.0))
    # blur_rain_mask=rain_mask
    image_RGB = np.array(image_RGB) / 255.
    means = np.mean(image_RGB, axis=(0, 1), keepdims=True)
    image_RGB = np.array(np.clip((image_RGB - means) * darken + means, 0, 1) * 255, dtype=np.uint8)

    return image_RGB

##rain_type='drizzle','heavy','torrential'
def rain(image, severity=2):  ## (200,200,200) a shade of gray
    # verify_image(image)
    image = np.asarray(image)
    slant = -1
    drop_length = 20
    drop_width = 1
    drop_color = (220, 220, 220)
    rain_type = severity
    darken_coefficient = [0.8, 0.8, 0.7, 0.6, 0.5]
    slant_extreme = slant

    imshape = image.shape
    if slant_extreme == -1:
        slant = np.random.randint(-10, 10)  ##generate random slant if no slant value is given
    rain_drops, drop_length = generate_random_lines(imshape, slant, drop_length, rain_type)
    output = rain_process(image, slant_extreme, drop_length, drop_color, drop_width, rain_drops,
                          darken_coefficient[severity - 1])
    image_RGB = output

    return image_RGB
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()
def fog(x,severity):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

    fog_layer = c[0] * plasma_fractal(wibbledecay=c[1])
    fog_clip=[]
    for i,image in enumerate(x):
        image = np.array(image) / 255.
        max_val = image.max()
        width,height,depth=image.shape
        image=image[width//2-112:width//2+112,height//2-112:height//2+112]
        image += np.repeat(fog_layer[:224, :224][..., np.newaxis],3,axis=2)
        fog_image = np.array(np.clip(image * max_val / (max_val + c[0]), 0, 1) * 255,dtype=np.uint8)
        fog_clip.append(fog_image)

    return fog_clip
def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = hsv2rgb(x)

    return np.clip(x, 0, 1) * 255
def motion_blur(x,severity=3):

    # motion_overlapping_frames=[3,5,7,9,11]
    motion_overlapping_frames=[1,2,3,4,6]
    c=motion_overlapping_frames[severity-1]

    clip=np.asarray(x)
    blur_clip=[]
    for i in range(c,clip.shape[0]-c):
        blur_image=np.sum(clip[i-c:i+c],axis=0,dtype=np.float)/(2.0*c)
        blur_clip.append(np.array(blur_image,dtype=np.uint8))
    return blur_clip
def contrast(x, severity=1):
    # c = [0.4, .3, .2, .1, .05][severity - 1]
    c = [0.5, 0.4, .3, .2, .1][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255
def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = hsv2rgb(x)

    return np.clip(x, 0, 1) * 255

#Image List
class RAVARDataset(Dataset):
    def __init__(
        self, training='train'):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.training = training
        self.tokenizer =  ClipTokenizer()
        self.annotation = []
        if training == 'train':
            f = open('/hkfs/work/workspace/scratch/fy2374-workspace/ijcai_folders/ravar/data_preprocessing/train_set_keyframe_crr_final.pkl', 'rb')
            self.key_bboxes = pkl.load(f)
            f.close()
            f = open('/hkfs/work/workspace/scratch/fy2374-workspace/ijcai_folders/ravar/data_preprocessing/train_set_label_crr_final.pkl', 'rb')
            self.annotations_doc = pkl.load(f)
            f.close()
            f = open('/hkfs/work/workspace/scratch/fy2374-workspace/ijcai_folders/ravar/data_preprocessing/train_set_description_crr_final.pkl','rb')
            self.captions = pkl.load(f)
            f.close()
        elif training == 'test':
            f = open('/hkfs/work/workspace/scratch/fy2374-workspace/ijcai_folders/ravar/data_preprocessing/test_set_keyframe_crr_final.pkl', 'rb')
            self.key_bboxes = pkl.load(f)
            f.close()
            f = open('/hkfs/work/workspace/scratch/fy2374-workspace/ijcai_folders/ravar/data_preprocessing/test_set_label_crr_final.pkl', 'rb')
            self.annotations_doc = pkl.load(f)
            f.close()
            f = open('/hkfs/work/workspace/scratch/fy2374-workspace/ijcai_folders/ravar/data_preprocessing/test_set_description_crr_final.pkl','rb')
            self.captions = pkl.load(f)
            f.close()      
        else:
            f = open('/hkfs/work/workspace/scratch/fy2374-workspace/ijcai_folders/ravar/data_preprocessing/val_set_keyframe_crr_final.pkl', 'rb')
            self.key_bboxes = pkl.load(f)
            f.close()
            f = open('/hkfs/work/workspace/scratch/fy2374-workspace/ijcai_folders/ravar/data_preprocessing/val_set_label_crr_final.pkl', 'rb')
            self.annotations_doc = pkl.load(f)
            f.close()
            f = open('/hkfs/work/workspace/scratch/fy2374-workspace/ijcai_folders/ravar/data_preprocessing/val_set_description_crr_final.pkl','rb')
            self.captions = pkl.load(f)
            f.close()                   
        mlb = MultiLabelBinarizer()
        self.keys = list(self.annotations_doc.keys())
        self.video_names = [item.split('___')[0] for item in self.keys]
        self.start_frames = [item.split('___')[1] for item in self.keys]
        self.perosn_ids = [item.split('___')[-1] for item in self.keys]
        self.key_bboxes = list(self.key_bboxes.values())
        self.annotations = list(self.annotations_doc.values())
        self.captions = list(self.captions.values())
        self.max_words = 32
        self.slice_framepos = 0
        self.max_frames = 8
        self.transform = self._transform([224,224])
        self.transform_key = self._transform_key()
        self.frame_order = 0
        self.annotations.append([item+1 for item in range(80)])
        self.annotations = mlb.fit_transform(np.array(self.annotations))
        self.annotations = self.annotations[:-1]
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}        
    def __len__(self):
        return len(self.annotations)

    def _get_text(self, caption):
        #caption = self.pseudo_caption_dict[pseudo_video_id]
        k = 1
        starts = np.zeros(k, dtype=np.long)
        ends = np.zeros(k, dtype=np.long)
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        for i in range(k):
            words = self.tokenizer.tokenize(caption)
            starts[i], ends[i] = 0, len(caption.split(' '))
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
        return caption, pairs_mask, pairs_segment, starts, ends

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]
        return raw_video_data

    def _get_rawvideo(self, frames):
        e = [len(frames)-1]
        s = [0]
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(s)
        # Pair x L x T x 3 x H x W
        video = np.zeros((len(s), self.max_frames, 3,
                          frames[0].shape[-2],frames[0].shape[-1]), dtype=np.float)
        try:
            for i in range(len(s)):
                start_time = int(s[i])
                end_time = int(e[i])
                start_time = start_time if start_time >= 0. else 0.
                end_time = end_time if end_time >= 0. else 0.
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = end_time + 1

                raw_video_data = np.stack(frames) #raw_video_data['video']

                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    # L x T x 3 x H x W
                    raw_video_slice = raw_video_data #self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                    if self.max_frames < raw_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = raw_video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = raw_video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                            video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    video_slice = self.process_frame_order(video_slice, frame_order=self.frame_order)
                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                    if slice_len < 1:
                        pass
                    else:
                        video[i][:slice_len, ...] = video_slice
                else:
                    print("video path: {} error. video id: {}, start: {}, end: {}".format(video_path, idx, start_time, end_time))
        except Exception as excep:
            print("video path: {} error. video id: {}, start: {}, end: {}, Error: {}".format(video_path, idx, s, e, excep))
            raise excep

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    def _transform_key(self,n_px=[340,465]):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            lambda image: image.convert("RGB"),
            ToTensor(),
        ])
    def video_to_tensor(self, frames, preprocess, sample_fp=0, start_time=None, end_time=None):
        images = []
        #frames_ = [rain(f) for f in frames[0]]
        for frame in frames[0]:
            #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (1,2,0))* 255
            images.append(preprocess(Image.fromarray(frame.astype(np.uint8)).convert("RGB")))
        if len(images) > 0:
            video_data = torch.tensor(np.stack(images))
        else:
            video_data = torch.zeros(1)
        return video_data

    def _load_imgs(self, start_frame, video_name):
        start_id = (int(start_frame)-900)*30 -45
        #print(start_id)
        images = []
        #tracker = MultiObjectTracker(dt=0.1)
        for i in range(90):
            formatted_id = '{:06}'.format((int(start_id)))
            img_path = '/hkfs/work/workspace/scratch/fy2374-workspace/ijcai_folders/ravar/ava_rgb_frames2/frames/'+video_name+'/image_'+formatted_id + '.jpg'
            image = cv2.imread(img_path)
            #images.append(image)
            #image = image + np.random.normal(0, 0.2, image.shape) 
            images.append(image)
        #images = shot_noise(np.stack(images),4) #motion_b, shot_noise, fog, rain
        return images

    def generate_samples(self,  final_imgs):

        imgs = []
        for img in final_imgs:
            imgs.append(img)
        return  imgs

    def generate_eval_clips(self, imgs):
        imgs = np.stack(imgs)
        clips = []
        length = len(imgs)
        pos = length/self.max_frames
        if length > self.max_frames:
            for start in range(int(pos)):
                sample_indx = np.linspace(start, len(imgs) - 1, num=self.max_frames, dtype=int)
                clips.append(imgs[sample_indx,...])
        return clips

    def __getitem__(self, index):
        if self.training == 'train':
            start_frame = self.start_frames[index]
            ann = self.annotations[index]
            video_name = self.video_names[index]
            
            imgs = self._load_imgs(start_frame, video_name)
            height = imgs[0].shape[0] 
            width = imgs[0].shape[1]
            # TODO this assumes image input, not general enough
            '''img_file = '{}.jpg'.format(ann["image_id"])
            image_path = os.path.join(self.vis_root, img_file)
            image = Image.open(image_path).convert("RGB")'''
            #imgs = np.stack(imgs)
            final_imgs = []
            for img in imgs:
                final_imgs.append(np.transpose(img, (2,0,1)))
            #tracking_bboxes, final_imgs = self.generate_samples(tracking_bboxes, final_imgs)
            #print(self.captions)
            key_frame = [[final_imgs[45]]]
            key_frame = self.video_to_tensor(key_frame, self.transform_key)
            bboxes = np.array(self.key_bboxes[index]) #np.array(tracking_bboxes)
            bboxes = bboxes[0]
            #print(bboxes)
            bboxes = np.array([bboxes[0], bboxes[1], bboxes[2], bboxes[3]])
            caption = self.captions[index][0]

            pairs_text, pairs_mask, pairs_segment, starts, ends = self._get_text(caption)
            video, video_mask = self._get_rawvideo(final_imgs)
            video = self.video_to_tensor(video, self.transform)
            #print(pairs_text)
            #print(ann)
            ann = np.array(ann)
            bboxes = np.array(bboxes).astype(np.float64)
            #print(bboxes)
            return  key_frame, pairs_text, pairs_mask, pairs_segment, video, video_mask, bboxes, ann
        else:
            start_frame = self.start_frames[index]
            ann = self.annotations[index]
            video_name = self.video_names[index]
            imgs = self._load_imgs(start_frame, video_name)
            height = imgs[0].shape[0] 
            width = imgs[0].shape[1]
            final_imgs = []
            pairs_texts, pairs_masks, pairs_segments, videos, video_masks, bboxess, anns = [],[],[],[],[],[],[]
            for img in imgs:
                final_imgs.append(np.transpose(img, (2,0,1)))
            key_frame = [[final_imgs[45]]]
            key_frame = self.video_to_tensor(key_frame, self.transform_key)
            #tracking_bboxes, final_imgs = self.generate_samples(tracking_bboxes, final_imgs)
            clips = self.generate_eval_clips(final_imgs)
            for clip in clips:
                caption = self.captions[index][0]
                pairs_text, pairs_mask, pairs_segment, starts, ends = self._get_text(caption)
                video, video_mask = self._get_rawvideo(clip)
                video = self.video_to_tensor(video, self.transform)
                pairs_texts.append(pairs_text)
                pairs_masks.append(pairs_mask)
                pairs_segments.append(pairs_segment)
                videos.append(video)
                video_masks.append(video_mask)
                anns.append(ann)
            #pairs_texts = np.stack(pairs_text)
            pairs_masks = np.stack(pairs_mask)
            pairs_segments = np.stack(pairs_segments)
            videos = torch.stack(videos)
            video_masks = np.stack(video_masks)
            #bboxess = np.array(self.key_bboxes[index]) #np.stack(bboxess)
            bboxes = np.array(self.key_bboxes[index]) #np.array(tracking_bboxes)
            bboxes = bboxes[0]
            bboxes = np.array([bboxes[0], bboxes[1], bboxes[2], bboxes[3]]).astype(np.float64)
            anns = np.stack(anns)


        return  key_frame, pairs_text, pairs_masks, pairs_segments, videos, video_masks, bboxes, anns

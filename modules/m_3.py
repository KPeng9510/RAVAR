from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch.distributed as dist
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from lavis.models import load_model_and_preprocess
from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from modules.modeling import CLIP4ClipPreTrainedModel, show_log, update_attr, check_attr
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

logger = logging.getLogger(__name__)
allgather = AllGather.apply
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)
from detr.models.backbone import Backbone, Joiner
from detr.models.detr import DETR, PostProcess
from detr.models.position_encoding import PositionEmbeddingSine
from detr.models.segmentation import DETRsegm, PostProcessPanoptic
from detr.models.transformer import Transformer
logger = logging.getLogger(__name__)
allgather = AllGather.apply

import pickle as pkl
f = open("/hkfs/work/workspace/scratch/fy2374-ijcai/ravar/benchmarks/exp_new_model/try_2_2_sem/modules/worlds_feature.pkl", "rb")
worlds_feature = pkl.load(f)
f.close()



def _make_detr(backbone_name: str, dilation=False, num_classes=91, mask=False):
    hidden_dim = 256
    backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=mask, dilation=dilation)
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True)
    detr = DETR(backbone_with_pos_enc, transformer, num_classes=num_classes, num_queries=100)
    if mask:
        return DETRsegm(detr)
    return detr
def detr_resnet50(pretrained=False, num_classes=91, return_postprocessor=False):
    """
    DETR R50 with 6 encoder and 6 decoder layers.

    Achieves 42/62.4 AP/AP50 on COCO val5k.
    """
    model = _make_detr("resnet50", dilation=False, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model
class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=4, window=14, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = window

        #self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
        #                     padding=1, groups=dim)
        '''self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, window))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, window))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
        self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
        self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))'''
        '''trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        trunc_normal_(self.ac_bias, std=.02)
        trunc_normal_(self.ca_bias, std=.02)'''
        pool_size = int(agent_num ** 0.5)
        #self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.pool = nn.Sequential(nn.Linear(dim, int(dim//2)), nn.GELU(), nn.Linear(int(dim//2), self.num_heads*self.agent_num*dim))
    def forward(self, x, attn_1=None, attn_2=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).contiguous().reshape(b, n, 3, c).contiguous().permute(2, 0, 1, 3).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v: b, n, c
        agent_tokens = self.pool(q)
        #print(agent_tokens.shape)
        q = q.reshape(b, n, num_heads, head_dim).contiguous().permute(0, 2, 1, 3).contiguous()
        k = k.reshape(b, n, num_heads, head_dim).contiguous().permute(0, 2, 1, 3).contiguous()
        v = v.reshape(b, n, num_heads, head_dim).contiguous().permute(0, 2, 1, 3).contiguous()
        agent_tokens = agent_tokens.mean(1).contiguous().reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        #position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window), mode='bilinear')
        #position_bias1 = position_bias1.contiguous().reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1).contiguous()
        #position_bias2 = (self.ah_bias + self.aw_bias).contiguous().reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1).contiguous()
        #position_bias = position_bias1 + position_bias2
        #position_bias = torch.cat([self.ac_bias.repeat(b, 1, 1, 1), position_bias], dim=-1)
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.contiguous().transpose(-2, -1).contiguous())
        agent_rep = agent_attn
        if attn_1 != None and attn_2 != None:
            agent_attn = agent_attn + 0.1*attn_1 + 0.1*attn_2
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v
        #agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        #agent_bias1 = agent_bias1.contiguous().reshape(1, num_heads, self.agent_num, -1).contiguous().permute(0, 1, 3, 2).contiguous().repeat(b, 1, 1, 1).contiguous()
        #agent_bias2 = (self.ha_bias + self.wa_bias).contiguous().reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1).contiguous()
        #agent_bias = agent_bias1 + agent_bias2
        #agent_bias = torch.cat([self.ca_bias.repeat(b, 1, 1, 1), agent_bias], dim=-2)
        #print((q * self.scale) @ agent_tokens.contiguous().transpose(-2, -1).shape)
        q_attn = self.softmax((q * self.scale) @ agent_tokens.contiguous().transpose(-2, -1))
        
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v
        x = x.contiguous().transpose(1, 2).reshape(b, n, c)
        #v_ = v.contiguous().transpose(1, 2).contiguous().reshape(b, h, w, c).contiguous().permute(0, 3, 1, 2).contiguous()
        x = x #+ self.dwc(v_).contiguous().permute(0, 2, 3, 1).contiguous().reshape(b, n - 1, c).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, agent_rep


class AgentBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.0, attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 agent_num=4, window=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AgentAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                   agent_num=agent_num, window=window)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_1=None, attn_2=None):
        att_x, att = self.attn(self.norm1(x), attn_1, attn_2)
        x = x + self.drop_path(att_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, att

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


@torch.no_grad()
def gather_together(data): #封装成一个函数，，用于收集各个gpu上的data数据，并返回一个list
    dist.barrier()
    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def gather(keys):
    # ....前面可以忽略，这里的keys可以看作negative_feat()
    keys = keys.detach().clone().cpu() # 先把数据移到cpu上
    gathered_list = gather_together(keys) # 进行汇总，得到一个list
    keys = torch.cat(gathered_list, dim=0).cuda()
    return keys 
    dist.gather(tensor, dst=root, group=group)

class Blipv2(nn.Module):
    def __init__(self,):
        super(Blipv2, self).__init__()
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=False)

        self.proj = nn.Linear(772,768)
        self.agent_temporal = AgentBlock(dim=768, num_heads=1, window=8) #AgentAttention()

        #self.bbox_regressio_head = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Linear(512,128), nn.ReLU(), nn.Linear(128,4))
        #self.classification_head = nn.Sequential(nn.Linear(768, 512), nn.Dropout(0.2), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 80))
        self.bbox_regressio_head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256,4))
        self.classification_head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 80))
        self.detection_model =  detr_resnet50(pretrained=True).eval()#torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True, force_reload=True)
        for param in self.detection_model.parameters():
            param.requires_grad = False
        self.proj_box = nn.Linear(100,8)
        self.agent_box = AgentBlock(dim=768, num_heads=1, window=8)
        self.bceloss = torch.nn.BCEWithLogitsLoss()
        self.mseloss = torch.nn.MSELoss()
        self.agent_semantic = AgentBlock(dim=768, num_heads=1, window=8)

    def forward(self, key_frame, input_ids, token_type_ids, attention_mask, video, video_mask=None, bbox=None, ann=None, training=True):
        #print('test')
        #i#nput_ids = input_ids.view(-1, input_ids.shape[-1])
        # T x 3 x H x W
        #print(key_frame.shape)
        worlds = worlds_feature.cuda()
        
        detect_results = self.detection_model(key_frame.squeeze())

        preds = detect_results["pred_logits"] # B,Q,N
        boxes = detect_results["pred_boxes"] # B,Q,N
        B = preds.shape[0]

        categories = torch.argmax(torch.nn.functional.softmax(preds,-1),-1).flatten(0,1)-1

        w_embs = torch.index_select(worlds, 0, categories)
        w_embs = w_embs.contiguous().view(B, 100, -1)
        #boxes_person = []
        '''for pred, box in zip(preds, boxes):
            mask = torch.argmax(torch.nn.functional.softmax(pred,-1),-1) == 1
            if True in mask:
                box_person = box[mask,:].contiguous().view(-1, 4) # B,Q,N
                boxes_person.append(box_person)
            else:
                boxes_person.append(torch.zeros(1, 4).cuda())
        #print(boxes_person.shape)
        #boxes_person = torch.cat(boxes_person)
        cls_scores = torch.max(preds[mask], -1)[0]
        batch,q,channels = boxes.shape
        box_container = torch.zeros(batch, 20, channels).cuda()
        if boxes_person.shape[1] <= 20:
            box_container[:,:boxes_person.shape[1],:] = boxes_person
        else:
            indexs = torch.topk(cls_scores,20,1)[1]
            resort_boxes = torch.index_select(boxes_person, 1, indexs)
            box_container = resort_boxes
        boxes = box_container'''
        # boxes = boxes.unsqueeze(1).repeat(1, 100, 1, 1)
        video = torch.as_tensor(video).float()
        video = video.unsqueeze(1).unsqueeze(1)
    
        b, pair, bs, ts, channel, h, w = video.shape
        '''agg_token_tem = self.aggregation_token_temporal.unsqueeze(1).repeat(b,1,1)
        agg_token_spa = self.aggregation_token_spatio.unsqueeze(1).repeat(b,1,1)
        agg_token_sem = self.aggregation_token_semantic.unsqueeze(1).repeat(b,1,1)
        agg_token_box = self.aggregation_token_boxes.unsqueeze(1).repeat(b,1,1)'''
        #print(video.shape)
        #print(boxes.shape)
        boxes = torch.stack([boxes[...,0] - 0.5*boxes[...,3], boxes[...,1] - 0.5*boxes[...,2], boxes[...,0] + 0.5*boxes[...,3],boxes[...,1] + 0.5*boxes[...,2]],-1)

        bboxes = self.proj(torch.cat([boxes, w_embs], -1))
        mask = torch.argmax(preds,-1) != 1
        bboxes[mask] = 0.0* bboxes[mask]
        #bboxes = self.proj_box(bboxes.contiguous().permute(0,2,1)).permute(0,2,1)
        '''bboxes = bboxes.unsqueeze(1).repeat(1, 8,1,1).flatten(1,2)
        boxes = torch.cat([bboxes, agg_token_box], dim=1)'''
        
        #f_boxes, attn_box = self.agent_box(boxes)
        #f_boxes = f_boxes.mean(1).unsqueeze(1) 
        video = video.contiguous().view(b * pair * bs* ts, channel, h, w)
        video_frame = bs * ts
        #print(input_ids)
        text = []
        for i in range(b):
            for j in range(ts):
                text.append(input_ids[i])
        sample = {"image": video.half().cuda(), "text_input": text}
        features_image = self.model.extract_features(sample, mode="image") # batch_size, 8, 32, 768
        features_text = self.model.extract_features(sample, mode="text") # batch_size, 12, 768
        features_text = features_text.text_embeds.contiguous().view(b, ts, -1, 768).mean(1)
        features_image = features_image.image_embeds.contiguous().view(b, ts, -1, 768)
        features_image,_ = self.agent_temporal(features_image.mean(-2))
        #print(features_image.shape)
        boxes = self.proj_box(boxes.contiguous().permute(0,2,1)).contiguous().permute(0,2,1)
        boxes,_ = self.agent_box(bboxes)
        features_text,_ = self.agent_semantic(features_text)
        logits = bboxes.mean(1) + features_text.mean(1) + features_image.mean(1)
        #logits = f_spatio.mean(1) + f_temporal.mean(1) + f_semantic.mean(1)#features_image.image_embeds.mean(1).contiguous().view(b, ts, -1).mean(1) + features_text.text_embeds.mean(1).contiguous().view(b, ts, -1).mean(1)

        bbox_prediction = self.bbox_regressio_head(logits)
        predictions = self.classification_head(logits)
        #print(predictions[0])
        #print(ann[0])
        if training:
            #predictions = gather(predictions)
            #ann = gather(ann)
            loss_cls = self.bceloss(predictions, ann.float().cuda())
            loss_cls = loss_cls
            batch_size = bbox_prediction.shape[0]
            loss_bbox = self.mseloss(bbox_prediction.reshape(batch_size, 4), bbox.float().cuda())
            
            loss = torch.mean(loss_bbox) + 5*torch.mean(loss_cls)
        else:
            loss = 0.0
        return loss, predictions, bbox_prediction, ann


def kernel(X, sigma):
    X = X.view(len(X), -1)
    XX = X @ X.t()
    X_sqnorms = torch.diag(XX)
    X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
    gamma = 1 / (2 * sigma ** 2)
    kernel_XX = torch.exp(-gamma * X_L2)
    return kernel_XX


def hsic_loss(input1, input2, unbiased=False, alternative=True):
    N = len(input1)
    if N < 4:
        return torch.tensor(0.0).to(input1.device)
    # we simply use the squared dimension of feature as the sigma for RBF kernel
    sigma_x = np.sqrt(input1.size()[1])
    sigma_y = np.sqrt(input2.size()[1])

    # compute the kernels
    kernel_XX = kernel(input1, sigma_x)
    kernel_YY = kernel(input2, sigma_y)

    if unbiased:
        """
        Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)
        hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )
        if alternative:
            loss = hsic
        else:
            loss = hsic / (N * (N - 3))
    else:
        """
        Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
        KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
        LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
        loss = torch.trace(KH @ LH / (N - 1) ** 2)
    if torch.isnan(loss):
        print("Capture hsic is nan.")
        loss = torch.tensor(0.0).to(input1.device)

    return loss


def kl_divergence(alpha, num_classes, device=None):
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def kl_loss(outputs, targets):
    evidence = F.relu(outputs)
    alpha = evidence + 1
    kl_alpha = (alpha - 1) * (1 - targets) + 1
    kl_loss = torch.mean(kl_divergence(kl_alpha, targets.shape[1], device=outputs.device))
    return kl_loss


# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
from torchvision.ops import nms
from imageio import imread
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.rpn.bbox_transform import clip_boxes
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections, _get_image_blob
from model.utils.parser_func import parse_args, set_dataset_args

from model.faster_rcnn.resnet import resnet
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY



if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)
  args = set_dataset_args(args, test=True)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  pascal_classes = np.asarray(['__background__','person'])

  print(args.class_agnostic)
  # initilize the network here.
  if args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res18':
    fasterRCNN = resnet(pascal_classes, 18, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()
  print("load checkpoint %s" % (args.load_name))
  if args.cuda > 0:
    checkpoint = torch.load(args.load_name)
  else:
    checkpoint = torch.load(args.load_name, map_location=(lambda storage, loc: storage))
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  print('load model successfully!')


  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  with torch.no_grad():
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()

  fasterRCNN.eval()

  start = time.time()
  
  max_per_image = 100
  thresh = 0.05
  vis = True

  webcam_num = args.webcam_num
  # Set up webcam or get image directories
  if webcam_num >= 0 :
    cap = cv2.VideoCapture('./tr_video_9.mp4')
    num_images = 0
  else:
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)

  print('Loaded Photo: {} images.'.format(num_images))


  while (num_images >= 0):
      total_tic = time.time()
      if webcam_num == -1:
        num_images -= 1

      # Get image from the webcam
      if webcam_num >= 0:
        if not cap.isOpened():
          raise RuntimeError("Webcam could not open. Please check connection.")
        ret, frame = cap.read()
        im_in = np.array(frame)
      # Load the demo image
      else:
        im_file = os.path.join(args.image_dir, imglist[num_images])
        # im = cv2.imread(im_file)
        im_in = np.array(imread(im_file))
        if len(im_in.shape) == 2:
          im_in = im_in[:,:,np.newaxis]
          im_in = np.concatenate((im_in,im_in,im_in), axis=2)
        # rgb -> bgr
        im_in = im_in[:,:,::-1]
      im = im_in

      blobs, im_scales = _get_image_blob(im)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)
      im_info_pt = torch.from_numpy(im_info_np)

      im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
      im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
      gt_boxes.resize_(1, 1, 5).zero_()
      num_boxes.resize_(1).zero_()

      # pdb.set_trace()
      det_tic = time.time()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
          pred_boxes = _.cuda() if args.cuda > 0 else _

      pred_boxes /= im_scales[0]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im2show = np.copy(im)
      for j in xrange(1, len(pascal_classes)):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            #keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            keep = keep = nms(cls_boxes[order, :],
                           cls_scores[order], 0.3)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
              im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      if webcam_num == -1:
          sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                           .format(num_images + 1, len(imglist), detect_time, nms_time))
          sys.stdout.flush()

      if vis and webcam_num == -1:
          # cv2.imshow('test', im2show)
          # cv2.waitKey(0)
          result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
          cv2.imwrite(result_path, im2show)
      else:
          cv2.imshow("frame", im2show)
          total_toc = time.time()
          total_time = total_toc - total_tic
          frame_rate = 1 / total_time
          print('Frame rate:', frame_rate)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
  if webcam_num >= 0:
      cap.release()
      cv2.destroyAllWindows()

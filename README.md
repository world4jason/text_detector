# Text Detector for OCR


This text detector acts as text localization and uses the structure of [RetinaNet](https://arxiv.org/pdf/1708.02002.pdf) and applies the techniques used in [textboxes++](https://arxiv.org/pdf/1801.02765.pdf).

## Train
```
cd text_detector/sample/SynthText
python3 train.py
```

---

## [TextBoxes++](https://arxiv.org/pdf/1801.02765.pdf)
- SSD structure is used, and vertical offset is added to make bbox proposal.
- The structure is the same as TextBoxes, but the offset for the QuadBox has been added.
- 4d-anchor box(xywh) offset -> (4+8)-d anchor box(xywh + x0y0x1y1x2y2x3y3) offset
- last conv : 3x5 -> To have a receptive field optimized for the quad box

## [RetinaNet](https://arxiv.org/pdf/1708.02002.pdf)
- Simple one-stage object detection and good performance
- FPN (Feature Pyramid Network) allows various levels of features to be used.
- output : 1-d score + 4-d anchor box offset
- cls loss = focal loss, loc loss = smooth L1 loss


## Encode
1. Define anchor boxes for each grid.
2. Obtain the IoU between the GT box and the anchor box.
3. Each anchor box is assigned to the largest GT box with IoU.
4. At this time, IoU> 0.5: Text (label = 1) / 0.4 <IoU <0.5: Ignore (label = -1) / IoU <0.4: non-text (label = 0).

## Todo list:
- Make Data
    - [x] make SynthText tfrecord
    - [x] make ICDAR13 tfrecord
    - [x] make ICDAR15 tfrecord
- Network
   - [x] ResNet
   - [x] Feature Pyramid Network
   - [x] Task Specific Network
   - [ ] Trainable BatchNorm (?
   - [ ] Freeze BatchNorm (?
   - [x] GroupNorm
   - [x] (binary) focal loss
- Utils
   - [x] add vertical offset
   - [x] validation infernece image visualization using Tensorboard
   - [ ] add augmentation ( + random crop)
   - [x] add evaluation code (mAP)
   - [x] QUAD version NMS (numpy version)

## Environment

- os : Ubuntu 16.04.4 LTS <br>
- GPU : Nvidia GTX 1080ti (12GB) <br>
- Python : 3.6.6 <br>
- Tensorflow : 1.4.0


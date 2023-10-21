# 👀*SEEM:* Segment Everything Everywhere All at Once

We introduce **SEEM** that can **S**egment **E**verything **E**verywhere with **M**ulti-modal prompts all at once. SEEM allows users to easily segment an image using prompts of different types including visual prompts (points, marks, boxes, scribbles and image segments) and language prompts (text and audio), etc. It can also work with any combinations of prompts or generalize to custom prompts!

:grapes: \[[Read our arXiv Paper](https://arxiv.org/pdf/2304.06718.pdf)\] &nbsp; :apple: \[[Try Hugging Face Demo](https://huggingface.co/spaces/xdecoder/SEEM)\] 

**One-Line Getting Started with Linux:**
```sh
git clone git@github.com:UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git && cd Segment-Everything-Everywhere-All-At-Once/demo_code && sh run_demo.sh
```

:point_right: *[New]* **Latest Checkpoints and Numbers:**
|                 |                                                                                             |          | COCO |      |      | Ref-COCOg |      |      | VOC   |       | SBD   |       |
|-----------------|---------------------------------------------------------------------------------------------|----------|------|------|------|-----------|------|------|-------|-------|-------|-------|
| Method          | Checkpoint                                                                                  | backbone | PQ   | mAP  | mIoU | cIoU      | mIoU | AP50 | NoC85 | NoC90 | NoC85 | NoC90 |
| X-Decoder       | [ckpt](https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focalt_last.pt) | Focal-T  | 50.8 | 39.5 | 62.4 | 57.6      | 63.2 | 71.6 | -     | -     | -     | -     |
| X-Decoder-oq201 | [ckpt](https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focall_last.pt) | Focal-L  | 56.5 | 46.7 | 67.2 | 62.8      | 67.5 | 76.3 | -     | -     | -     | -     |
| SEEM            | [ckpt](https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v2.pt)      | Focal-T  | 50.6 | 39.4 | 60.9 | 58.5      | 63.5 | 71.6 | 3.54  | 4.59  | *     | *     |
| SEEM            | -                                                                                           | Davit-d3 | 56.2 | 46.8 | 65.3 | 63.2      | 68.3 | 76.6 | 2.99  | 3.89  | 5.93  | 9.23  |
| SEEM-oq101      | [ckpt](https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt)       | Focal-L  | 56.2 | 46.4 | 65.5 | 62.8      | 67.7 | 76.2 | 3.04  | 3.85  | *     | *     |

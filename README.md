# Image Captioning with Meta-CLIP and MT5

Link dataset: [https://huggingface.co/datasets/Skyler215/KTVIC](https://huggingface.co/datasets/Skyler215/KTVIC)

## Results

| Model ("MetaCLIP size"-"mT5 size") | BLEU-1 | BLEU-4 | **CIDEr** | METEOR | ROUGE-L |
|-------|-----------|--------|--------|---------|--------|
| b16-small | 68.94 | 34.81 | 98.50 | 53.73 | 60.56 |
| l14-large | 69.98 | 36.77 | 103.12 | 55.19 | 62.19 |
| h14-large | 67.59 | 30.07 | 68.55 | 47.93 | 57.15 |

## Train model

**WARNING**: Training code hiện tại không chạy được vì train dataset không có preprocessed images. Code sẽ được cập nhật trong thời gian tới.

1. Tải KTVIC dataset
2. Cài đặt Java (nếu muốn evaluate CIDEr)
3. Cài đặt pytorch 2.5.1 theo hướng dẫn trên website
4. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```
5. Search `EDIT` trong file `clip_mt5_large_img_cap.py` và chỉnh sửa cho phù hợp
6. Chạy file
```bash
python clip_mt5_large_img_cap.py
```

## Evaluate model
Model sẽ được evaluate trên tập test của KTVIC dataset sau mỗi epoch và kết quả sẽ được in trên terminal.
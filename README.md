# Image Captioning with Meta-CLIP and MT5

Link dataset: [https://huggingface.co/datasets/Skyler215/KTVIC](https://huggingface.co/datasets/Skyler215/KTVIC)

## Train model
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
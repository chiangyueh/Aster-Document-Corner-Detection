# Aster — Document Corner Detection

## 安裝

```bash
cd Aster
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```


## 訓練

GPU
```bash
python -m aster train
```
CPU

```bash
python -m aster train --set train.device=cpu
```

## 評估

```bash
python -m aster evaluate --model weights/Aster.pt --data data/real_test --vis export/preview
```

## 匯出

```bash
python -m aster export --model weights/Aster.pt --out export/Aster.onnx --mnn export/Aster.mnn
```

原始照片：
![原始照片](examples/document_original.png)

模型推論後的四角標記：
![四角標記](examples/document_marked.png)

## Camonster 文件攤平效果

[Aster 模型應用於 Camonster 的實際文件攤平效果](https://www.camonster.online/)（縮小圖）。
![文件攤平效果](examples/document-after-small.webp)

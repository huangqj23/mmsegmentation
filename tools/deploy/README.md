# 导出到 T9（ONNX → TensorRT 10.13 / OpenVINO 2025.4）

一条命令导出一个分割模型，并在 T9 用的推理后端上对拍。不需要 mmdeploy。

```bash
python tools/deploy/export.py CONFIG CHECKPOINT --shape H W --out work_dirs/deploy/NAME \
    [--images 图片目录] [--head logits|prob|label] [--normalize none|graph]
```

产物：

| 文件 | 内容 |
|---|---|
| `NAME.onnx` | 静态尺寸，opset 17，只有标准算子 |
| `NAME.json` | 给 C++ 端：输入形状、通道顺序、mean / std、输出含义、类别、阈值、建引擎命令 |
| `NAME.report.json` | 检查结果和对拍数字 |
| `NAME.fp16.engine` | 本机建的 FP16 引擎，只用于检查；T9 要在部署机上自己建 |

## 和 T9 现网模型的接口一致

T9 的 `RecipeFiles_T9*` 里的模型都是这个样子，导出默认也是：

- 输入 `input1`，`[1, 3, H, W]` float32，归一化在 C++ 里做（`--normalize none`）。`NAME.json` 写明要用的 mean / std 和通道顺序。
- 输出 `output1`，`[1, C, H, W]` float32，已经缩放到输入尺寸。默认是 logits：二值分割（`out_channels=1`）就是单通道 logits，和现网一样；多类别是 C 个通道。`--head label` 输出每个像素的类别号（float32）。

## 工具做了什么

1. **池化尺寸检查**：`AdaptiveAvgPool2d` 的分箱除不尽特征图时导不出来，工具会提示能用的尺寸（PSPNet 取 48 的倍数，UPerNet 取 192 的倍数）。
2. **导出**：默认 TorchScript 导出器；`--dynamo` 用新导出器。
3. **图检查**：不能有自定义域节点（TensorRT 要插件、OpenVINO 跑不了）、不能有随机算子（SegNeXt 默认每次推理随机初始化，加 `--fix-random-bases`）、opset 不超过 17、尺寸全静态。
4. **对拍**：ONNX Runtime（CPU）、OpenVINO（CPU，强制 FP32）、TensorRT（FP32；FP16 和 `trtexec --fp16` 一样建），都和 PyTorch 比：输出最大误差、像素一致率（取各图平均）、以 PyTorch 结果为准的 mIoU。门槛：FP32 后端 ≥ 99.9%，TensorRT FP16 ≥ 99.5%（暂定，有了带标注的验证集后改成 mIoU 差）。
5. **时延**：TensorRT 引擎的 GPU 时间中位数，以及 PyTorch 的时间（同一张卡）。

二值分割的例子（MiT 主干，和 T9 的 SegFormer 模型同结构）：

```bash
python tools/deploy/export.py configs/segformer/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py \
    best.pth --shape 960 960 --out work_dirs/deploy/aa_seg \
    --cfg-options model.decode_head.out_channels=1 model.decode_head.threshold=0.5
```

## 到 T9 那边

```bat
trtexec.exe --onnx=NAME.onnx --saveEngine=NAME.engine --fp16
```

引擎和 GPU 型号、TensorRT 版本、操作系统绑定，服务器上建的引擎不能拿去用。

一次跑一批模型并汇总成表：`python tools/deploy/summarize.py work_dirs/deploy/*.report.json`。

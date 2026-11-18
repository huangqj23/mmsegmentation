# Copyright (c) OpenMMLab. All rights reserved.
"""Export a segmentor to ONNX for the T9 runtime and check it end to end.

One command does the whole chain; mmdeploy is not needed::

    python tools/deploy/export.py CONFIG CHECKPOINT --shape 512 1024 \
        --out work_dirs/deploy/NAME [--images DIR]

1. Build the model and wrap it: resize the logits to the input size, then
   the chosen head (``logits`` by default). The interface matches the models
   T9 runs today: input ``input1`` [1, 3, H, W] float32, output ``output1``
   [1, C, H, W] float32, static shape, normalization outside the graph
   (``--normalize graph`` moves it inside).
2. Check AdaptiveAvgPool2d bins divide their feature maps (the exporters
   reject them otherwise) and suggest input sizes that work.
3. Export ONNX (static shape, opset 17 by default) and check the graph: no
   custom domains, no random ops, opset within what the targets parse.
4. Run it in ONNX Runtime (CPU), OpenVINO (CPU, FP32) and TensorRT (FP32,
   and FP16 the way ``trtexec --fp16`` builds) on real images and compare
   with PyTorch: max |diff| of the output and pixel agreement of the
   predicted labels.
5. Write ``NAME.onnx``, ``NAME.json`` (what the C++ side has to know:
   input layout, channel order, mean / std, output meaning) and
   ``NAME.report.json``. The FP16 engine built here (``NAME.fp16.engine``)
   is for this machine only; T9 builds its own with ``trtexec.exe``.
"""
import argparse
import copy
import glob
import json
import math
import os
import os.path as osp
import platform
import statistics
import time
import traceback
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config, DictAction
from mmengine.model import revert_sync_batchnorm

from mmseg.apis import init_model

# what the deployment side parses: TensorRT 10.13 (T9) and OpenVINO 2025.4
MAX_OPSET = 17
INPUT_NAME, OUTPUT_NAME = 'input1', 'output1'
# provisional gates (mmcv plan, stage 1): pixel agreement with PyTorch
GATES = {'onnxruntime': 0.999, 'openvino': 0.999, 'tensorrt_fp32': 0.999,
         'tensorrt_fp16': 0.995}


def parse_args():
    parser = argparse.ArgumentParser(description='Export a segmentor for T9')
    parser.add_argument('config')
    parser.add_argument('checkpoint', nargs='?', default=None,
                        help='weights (path or URL); random weights if omitted')
    parser.add_argument('--shape', type=int, nargs=2, required=True,
                        metavar=('H', 'W'), help='static input size')
    parser.add_argument('--out', required=True,
                        help='output prefix, e.g. work_dirs/deploy/segformer_b0')
    parser.add_argument('--head', choices=['logits', 'prob', 'label'],
                        default='logits',
                        help='logits: raw scores (T9 today); prob: sigmoid / '
                        'softmax; label: class id per pixel, as float32')
    parser.add_argument('--normalize', choices=['none', 'graph'],
                        default='none',
                        help='none: the caller normalizes (T9 today); graph: '
                        'mean / std and channel order inside the graph, input '
                        'is the 0-255 BGR image')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--dynamo', action='store_true',
                        help='use the torch.export based exporter')
    parser.add_argument('--images', nargs='*', default=None,
                        help='image files or directories for the comparison '
                        '(default: the images in demo/)')
    parser.add_argument('--num-images', type=int, default=8)
    parser.add_argument('--fix-random-bases', action='store_true',
                        help='SegNeXt: fixed matrix-decomposition bases instead '
                        'of a random draw at every inference')
    parser.add_argument('--no-trt', action='store_true')
    parser.add_argument('--no-openvino', action='store_true')
    parser.add_argument('--device', default='cuda:0',
                        help='device for the PyTorch reference')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override config entries, e.g. model.decode_head.out_channels=1 '
        'for a binary (sigmoid) head like T9\'s models')
    return parser.parse_args()


class DeployWrapper(nn.Module):
    """Segmentor + resize to the input size + output head."""

    def __init__(self, segmentor, size, head, threshold=0.5, norm=None):
        super().__init__()
        self.segmentor = segmentor
        self.size = tuple(size)
        self.head = head
        self.threshold = threshold
        decode_head = segmentor.decode_head
        if isinstance(decode_head, nn.ModuleList):  # cascade (OCRNet)
            decode_head = decode_head[-1]
        self.align_corners = decode_head.align_corners
        self.norm = norm is not None
        if self.norm:
            mean, std, bgr_to_rgb = norm
            self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
            self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))
            self.bgr_to_rgb = bgr_to_rgb

    def forward(self, x):
        if self.norm:
            if self.bgr_to_rgb:
                x = x[:, [2, 1, 0]]
            x = (x - self.mean) / self.std
        logits = self.segmentor._forward(x)
        logits = F.interpolate(
            logits, size=self.size, mode='bilinear',
            align_corners=self.align_corners)
        if self.head == 'logits':
            return logits
        binary = logits.shape[1] == 1
        if self.head == 'prob':
            return logits.sigmoid() if binary else logits.softmax(dim=1)
        if binary:
            return (logits.sigmoid() > self.threshold).float()
        return logits.argmax(dim=1, keepdim=True).float()


def labels_of(out, head, threshold):
    """Per-pixel class ids from any head's output, shape (H, W)."""
    out = out[0]
    if head == 'label':
        return out[0].round().astype(np.int64)
    if out.shape[0] == 1:
        prob = out[0] if head == 'prob' else 1 / (1 + np.exp(-out[0]))
        return (prob > threshold).astype(np.int64)
    return out.argmax(axis=0)


def agreement(ref, other):
    """Pixel agreement and mIoU of `other` taking `ref` as ground truth."""
    acc = float((ref == other).mean())
    ious = []
    for c in np.union1d(np.unique(ref), np.unique(other)):
        inter = np.logical_and(ref == c, other == c).sum()
        union = np.logical_or(ref == c, other == c).sum()
        ious.append(inter / union)
    return acc, float(np.mean(ious))


def check_pooling(model, example):
    """AdaptiveAvgPool2d needs bins that divide the feature map to export."""
    records = []

    def hook(module, inputs, output):
        records.append((module.output_size, tuple(inputs[0].shape[-2:])))

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d))]
    with torch.no_grad():
        model(example)
    for h in handles:
        h.remove()
    H, W = example.shape[-2:]
    need_h = need_w = 1
    bad = []
    for out_size, (fh, fw) in records:
        oh, ow = (out_size, out_size) if isinstance(out_size, int) else out_size
        oh, ow = oh or fh, ow or fw
        if fh % oh or fw % ow:
            bad.append(f'bins {oh}x{ow} on {fh}x{fw}')
        need_h = math.lcm(need_h, H // fh * oh)
        need_w = math.lcm(need_w, W // fw * ow)
    suggest = None
    if bad:
        suggest = [max(need_h, round(H / need_h) * need_h),
                   max(need_w, round(W / need_w) * need_w)]
    return {'pools': len(records), 'bad': sorted(set(bad)),
            'multiple_of': [need_h, need_w], 'suggest': suggest}


def check_graph(path, opset):
    import onnx
    model = onnx.load(path)
    onnx.checker.check_model(model)
    graph = model.graph
    ops = Counter((n.domain or 'ai.onnx') + '::' + n.op_type for n in graph.node)
    custom = sorted(k for k in ops if not k.startswith('ai.onnx::'))
    random_ops = sorted(k for k in ops if k.split('::')[1].startswith('Random'))
    model_opset = max(o.version for o in model.opset_import
                      if o.domain in ('', 'ai.onnx'))
    inits = {i.name for i in graph.initializer}

    def io(v):
        t = v.type.tensor_type
        dims = [d.dim_value if d.HasField('dim_value') else d.dim_param
                for d in t.shape.dim]
        return {'name': v.name, 'shape': dims,
                'dtype': onnx.TensorProto.DataType.Name(t.elem_type).lower()}

    inputs = [io(v) for v in graph.input if v.name not in inits]
    outputs = [io(v) for v in graph.output]
    static = all(isinstance(d, int) and d > 0
                 for v in inputs + outputs for d in v['shape'])
    problems = []
    if custom:
        problems.append(f'custom-domain nodes: {custom}')
    if random_ops:
        problems.append(f'random ops (TensorRT ignores seeds): {random_ops}')
    if model_opset > MAX_OPSET:
        problems.append(f'opset {model_opset} > {MAX_OPSET}')
    if not static:
        problems.append('dynamic shapes')
    return {'nodes': len(graph.node), 'opset': model_opset,
            'ops': dict(sorted(ops.items())), 'inputs': inputs,
            'outputs': outputs, 'problems': problems}


def load_images(paths, num, shape, norm, normalize_in_graph):
    """Real images, resized to the export size, as the graph's input."""
    files = []
    for p in paths:
        if osp.isdir(p):
            for ext in ('jpg', 'jpeg', 'png', 'JPEG', 'bmp'):
                files += glob.glob(osp.join(p, '**', f'*.{ext}'),
                                   recursive=True)
        else:
            files.append(p)
    H, W = shape
    batch, used = [], []
    for f in sorted(set(files)):
        if len(used) == num:
            break
        img = cv2.imread(f, cv2.IMREAD_COLOR)  # BGR, like mmseg's loader
        if img is None:  # not an image cv2 can read
            continue
        used.append(f)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32)
        if not normalize_in_graph:
            mean, std, bgr_to_rgb = norm
            if bgr_to_rgb:
                x = x[..., ::-1]
            x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
        batch.append(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
    return used, batch


def run_onnxruntime(path, inputs):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    return [sess.run(None, {INPUT_NAME: x})[0] for x in inputs]


def run_openvino(path, inputs):
    import openvino as ov
    core = ov.Core()
    # compare in FP32: on CPUs with bf16 units OpenVINO would pick bf16
    compiled = core.compile_model(
        path, 'CPU', {'INFERENCE_PRECISION_HINT': 'f32'})
    return [compiled(x)[0] for x in inputs], ov.__version__


def build_trt(path, fp16, engine_path=None):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(path, 'rb') as f:
        if not parser.parse(f.read()):
            raise RuntimeError('; '.join(
                str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    if fp16:  # what `trtexec --fp16` does
        config.set_flag(trt.BuilderFlag.FP16)
    start = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('TensorRT engine build failed')
    build_s = time.time() - start
    if engine_path:
        with open(engine_path, 'wb') as f:
            f.write(plan)
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    return engine, build_s, trt.__version__


def run_trt(engine, inputs, warmup=10, iters=50):
    import tensorrt as trt
    ctx = engine.create_execution_context()
    dtype = {trt.float32: torch.float32, trt.float16: torch.float16,
             trt.int32: torch.int32, trt.int64: torch.int64}
    out_shape = tuple(engine.get_tensor_shape(OUTPUT_NAME))
    out = torch.empty(out_shape, device='cuda',
                      dtype=dtype[engine.get_tensor_dtype(OUTPUT_NAME)])
    stream = torch.cuda.Stream()
    results = []
    for x in inputs:
        inp = torch.from_numpy(x).cuda()
        ctx.set_tensor_address(INPUT_NAME, inp.data_ptr())
        ctx.set_tensor_address(OUTPUT_NAME, out.data_ptr())
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        results.append(out.float().cpu().numpy())
    # latency on the first input, GPU time per run
    times = []
    for i in range(warmup + iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        ctx.execute_async_v3(stream.cuda_stream)
        end.record(stream)
        stream.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    return results, statistics.median(times)


def run_torch(model, inputs, device, warmup=10, iters=50):
    model = copy.deepcopy(model).to(device).eval()
    results = []
    with torch.no_grad():
        for x in inputs:
            results.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
        x = torch.from_numpy(inputs[0]).to(device)
        times = []
        for i in range(warmup + iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(start.elapsed_time(end))
    return results, statistics.median(times)


def compare(name, ref, outs, head, threshold):
    diffs = [float(np.abs(r - o).max()) for r, o in zip(ref, outs)]
    accs, mious = zip(*(agreement(labels_of(r, head, threshold),
                                  labels_of(o, head, threshold))
                        for r, o in zip(ref, outs)))
    row = {'max_abs_diff': max(diffs),
           'pixel_agreement': float(np.mean(accs)),
           'min_pixel_agreement': min(accs),
           'miou_vs_torch': float(np.mean(mious)),
           'per_image': [round(a, 6) for a in accs]}
    # gated on the mean: out-of-domain images have many near-tied logits,
    # which FP16 flips, and say little about the deployed data
    row['pass'] = row['pixel_agreement'] >= GATES[name]
    return row


def main():
    args = parse_args()
    torch.manual_seed(0)
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    name = osp.basename(args.out)
    onnx_path = args.out + '.onnx'
    report = {'config': args.config, 'checkpoint': args.checkpoint,
              'shape': args.shape, 'head': args.head,
              'normalize': args.normalize, 'opset': args.opset,
              'exporter': 'dynamo' if args.dynamo else 'torchscript',
              'torch': torch.__version__, 'host': platform.node()}

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
        report['cfg_options'] = {k: str(v) for k, v in args.cfg_options.items()}
    model = init_model(cfg, args.checkpoint, device='cpu')
    model = revert_sync_batchnorm(model).eval()
    if args.fix_random_bases:
        for m in model.modules():
            if getattr(m, 'rand_init', False):
                m.rand_init = False  # registers fixed bases on first forward
    dp = model.cfg.model.data_preprocessor
    norm = (list(dp['mean']), list(dp['std']), bool(dp.get('bgr_to_rgb')))
    out_channels = model.decode_head[-1].out_channels if isinstance(
        model.decode_head, nn.ModuleList) else model.decode_head.out_channels
    threshold = getattr(model.decode_head, 'threshold', None) or 0.5
    wrapper = DeployWrapper(
        model, args.shape, args.head, threshold,
        norm if args.normalize == 'graph' else None).eval()

    H, W = args.shape
    if args.normalize == 'graph':
        example = torch.rand(1, 3, H, W) * 255
    else:
        example = torch.randn(1, 3, H, W)
    report['pooling'] = check_pooling(wrapper, example)
    if report['pooling']['bad']:
        p = report['pooling']
        report['status'] = 'fail: pooling'
        print(f"AdaptiveAvgPool2d {p['bad']} do not divide; input sizes must "
              f"be multiples of {p['multiple_of']}, e.g. --shape "
              f"{p['suggest'][0]} {p['suggest'][1]}")
        return finish(args, report)

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper, (example, ), onnx_path, input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME], opset_version=args.opset,
                do_constant_folding=True, dynamo=args.dynamo)
    except Exception as e:  # noqa: BLE001 - report every exporter failure
        report['status'] = 'fail: export'
        report['error'] = f'{type(e).__name__}: {e}'[:2000]
        traceback.print_exc()
        return finish(args, report)
    report['graph'] = check_graph(onnx_path, args.opset)
    problems = list(report['graph']['problems'])

    files, inputs = load_images(
        args.images or [osp.join(osp.dirname(__file__), '../../demo')],
        args.num_images, args.shape, norm, args.normalize == 'graph')
    if not inputs:
        inputs = [example.numpy()]
    report['images'] = [osp.basename(f) for f in files]

    ref, torch_ms = run_torch(wrapper, inputs, args.device)
    report['torch_ms'] = torch_ms
    backends = {}
    try:
        backends['onnxruntime'] = compare(
            'onnxruntime', ref, run_onnxruntime(onnx_path, inputs),
            args.head, threshold)
    except Exception as e:  # noqa: BLE001
        backends['onnxruntime'] = {'error': str(e)[:500], 'pass': False}
    if not args.no_openvino:
        try:
            outs, ov_version = run_openvino(onnx_path, inputs)
            backends['openvino'] = compare('openvino', ref, outs, args.head,
                                           threshold)
            report['openvino'] = ov_version
        except Exception as e:  # noqa: BLE001
            backends['openvino'] = {'error': str(e)[:500], 'pass': False}
    if not args.no_trt:
        for fp16 in (False, True):
            key = 'tensorrt_fp16' if fp16 else 'tensorrt_fp32'
            try:
                engine, build_s, trt_version = build_trt(
                    onnx_path, fp16, args.out + '.fp16.engine' if fp16 else None)
                outs, ms = run_trt(engine, inputs)
                backends[key] = compare(key, ref, outs, args.head, threshold)
                backends[key].update(build_s=round(build_s, 1), gpu_ms=ms)
                report['tensorrt'] = trt_version
            except Exception as e:  # noqa: BLE001
                backends[key] = {'error': str(e)[:1000], 'pass': False}
    report['backends'] = backends
    problems += [f'{k}: {v.get("error") or "below gate"}'
                 for k, v in backends.items() if not v['pass']]
    report['status'] = 'pass' if not problems else 'fail: ' + '; '.join(problems)

    write_manifest(args, name, report, norm, out_channels, threshold, model)
    return finish(args, report)


def write_manifest(args, name, report, norm, out_channels, threshold, model):
    """What the C++ side needs to feed the model and read its output."""
    mean, std, bgr_to_rgb = norm
    H, W = args.shape
    classes = list(model.dataset_meta.get('classes', [])) if getattr(
        model, 'dataset_meta', None) else []
    if args.normalize == 'graph':
        inp = {'pixels': 'BGR, 0-255, float32 (normalized inside the graph)'}
    else:
        inp = {'pixels': ('RGB' if bgr_to_rgb else 'BGR') +
               ', (x - mean) / std on 0-255 values',
               'mean': mean, 'std': std}
    head = {'logits': 'raw scores', 'prob': 'sigmoid / softmax',
            'label': 'class id per pixel (float32)'}[args.head]
    manifest = {
        'onnx': name + '.onnx',
        'opset': report['graph']['opset'],
        'input': {'name': INPUT_NAME, 'shape': [1, 3, H, W],
                  'dtype': 'float32', 'layout': 'NCHW', **inp},
        'output': {'name': OUTPUT_NAME,
                   'shape': [1, 1 if args.head == 'label' else out_channels,
                             H, W],
                   'dtype': 'float32', 'meaning': head,
                   'binary': out_channels == 1, 'threshold': threshold,
                   'classes': classes},
        'tensorrt': {'version': report.get('tensorrt'),
                     'build': f'trtexec --onnx={name}.onnx '
                     f'--saveEngine={name}.engine --fp16'},
        'source': {'config': args.config, 'checkpoint': args.checkpoint,
                   'torch': torch.__version__},
    }
    with open(args.out + '.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def finish(args, report):
    with open(args.out + '.report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    b = report.get('backends', {})
    cells = []
    for k, v in b.items():
        if 'error' in v:
            cells.append(f'{k}=ERROR')
        else:
            cells.append(f"{k}={v['pixel_agreement']:.4%}")
    trt16 = b.get('tensorrt_fp16', {}).get('gpu_ms')
    print(f"[{report['status']}] {osp.basename(args.out)} "
          f"nodes={report.get('graph', {}).get('nodes')} " + ' '.join(cells) +
          (f' trt16={trt16:.2f}ms' if trt16 else ''))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())

# Copyright (c) OpenMMLab. All rights reserved.
"""Collect export.py reports into one markdown table.

    python tools/deploy/summarize.py work_dirs/deploy/*.report.json
"""
import json
import os.path as osp
import sys


def pct(v):
    return '—' if v is None else f'{v:.2%}'


def main(paths):
    cols = ['onnxruntime', 'openvino', 'tensorrt_fp32', 'tensorrt_fp16']
    print('| model | input | status | nodes | ORT | OpenVINO | TRT FP32 | '
          'TRT FP16 | FP16 max diff | TRT FP16 ms | PyTorch ms |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for p in sorted(paths):
        with open(p, encoding='utf-8') as f:
            r = json.load(f)
        b = r.get('backends', {})
        cells = []
        for c in cols:
            v = b.get(c)
            cells.append('error' if v and 'error' in v else
                         pct(v.get('pixel_agreement')) if v else '—')
        fp16 = b.get('tensorrt_fp16', {})
        status = r['status'] if r['status'] == 'pass' else r['status'][:60]
        name = osp.basename(p).replace('.report.json', '')
        diff = fp16.get('max_abs_diff')
        print(f"| {name} | {r['shape'][0]}×{r['shape'][1]} | {status} | "
              f"{r.get('graph', {}).get('nodes', '—')} | " + ' | '.join(cells) +
              f" | {'—' if diff is None else f'{diff:.3g}'}"
              f" | {fp16.get('gpu_ms', 0):.2f} | {r.get('torch_ms', 0):.2f} |")


if __name__ == '__main__':
    main(sys.argv[1:])

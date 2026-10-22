# -*- coding: utf-8 -*-
from rknn.api import RKNN

rknn = RKNN(verbose=True)
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rk3588',
    quantized_dtype='asymmetric_quantized-8',
)
rknn.load_onnx(model='best.onnx')
rknn.build(do_quantization=True, dataset='dataset.txt')
rknn.export_rknn('best_int8_232.rknn')
rknn.release()
print("完成：best_int8_232.rknn")

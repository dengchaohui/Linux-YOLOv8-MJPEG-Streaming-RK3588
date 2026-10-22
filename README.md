# 🚀 RK3588 Linux YOLOv8 Digit Recognition & Real-time MJPEG Streaming

[![Platform](https://img.shields.io/badge/Platform-RK3588-blue)]()
[![OS](https://img.shields.io/badge/OS-Linux%206.1%20%7C%20Ubuntu%2022.04-orange)]()
[![Framework](https://img.shields.io/badge/Framework-RKNN%202.3.2-green)]()

An end-to-end Edge AI deployment project for YOLOv8 (10-class digits) on the RK3588 platform. Achieved camera capture, NPU inference, and wireless MJPEG streaming without a physical display.

![Real-time Demo](images/demo_result.jpg)

### ✨ Key Features
*   **9-Tensor Output Structure**: Solved the notorious "zero-class" bug caused by DFL fusion in the standard single-tensor YOLOv8 ONNX, by exporting the 9-tensor structure via Rockchip's fork.
*   **High-Precision Quantization**: Achieved an INT8 quantization cosine similarity of **0.996** across the entire network, maintaining FP32-level accuracy while accelerating NPU inference.
*   **Headless Wireless Streaming**: Implemented a custom HTTP MJPEG server using Python and GStreamer, bypassing V4L2 limitations for rkisp, enabling live streaming directly to a PC browser over LAN.
*   **Firmware-Level Delivery**: Packaged the AI scripts, models, systemd auto-start services, and dependencies into the Linux SDK rootfs (via Overlay mechanism), generating a "plug-and-play" `update.img`.

---

## 🇨🇳 中文简介 (Chinese Introduction)

本项目基于迅为 iTOP-RK3588 开发板，完整实现了 YOLOv8 数字识别从模型量化转换、GStreamer 推流到 SDK 固件级交付的全闭环。

### 💡 核心避坑点（全网独家）
1. **彻底解决单张量 `[1,14,8400]` 转 RKNN 导致类别分数全为 0 的死坑**（改用 Rockchip Fork 导出 9 张量 DFL 未融合结构）。
2. **解决无屏推流痛点**：绕过 OpenCV V4L2 限制，使用 GStreamer + MJPEG 实现局域网低延迟网页展示。
3. **系统级封装**：通过 Linux SDK Overlay 机制把 Python 脚本和 `.service` 服务编译进系统镜像，烧录即用。

### 📂 仓库目录
*   `docs/`：万字级底层排坑与部署文档（中文）。
*   `scripts/`：Python 端 MJPEG 推流推理源码。
*   `cpp_demo/`：C++ 端 9 张量后处理源码。
*   `sdk_overlay_patch/`：Linux SDK 增量编译补丁包。
*   `images/`：实机运行效果截图。

---

## 🛠️ Quick Start
1. Ensure your RK3588 board is running Linux 6.1 with `librknnrt.so` (v2.3.0).
2. Check the `/docs` folder for detailed Chinese deployment guides.
3. For Python streaming: 
   ```bash
   export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
   python3 rknn_infer_mjpeg.py

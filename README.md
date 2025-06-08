# kang
1. ccnet.py
CCNet（Criss-Cross Network）
引入了 Criss-Cross Attention Module，可在水平方向和垂直方向上进行上下文信息聚合，以减少全局自注意力的计算开销，从而提升分割精度，尤其适合处理空间结构复杂的图像。

2. cdunet.py
CD-UNet（Change Detection U-Net）
基于经典 UNet，针对变化检测任务进行优化，常用于遥感图像中前后时间序列的对比分析。它可能集成双分支结构来处理变化前后的图像对。

3. deeplabv2.py
DeepLabV2
提出 空洞卷积（Atrous Convolution） 与 多尺度空洞空间金字塔池化（ASPP），用于增强感受野并捕捉多尺度上下文信息，在保持分辨率的同时提取更丰富的特征。

4. deeplabv3plus.py
DeepLabV3+
是在 DeepLabV3 的基础上加入解码器模块，增强边界信息恢复能力，结合 ASPP 与 Encoder-Decoder 架构，是目前语义分割性能较强的模型之一。

5. enet.py
ENet（Efficient Neural Network）
轻量化的实时语义分割网络，参数量小、速度快，适合边缘设备或资源受限场景。它通过提前降采样和分组卷积等设计提高效率。

6. pspnet.py
PSPNet（Pyramid Scene Parsing Network）
通过 金字塔池化模块（Pyramid Pooling Module） 融合不同尺度的上下文信息，提升对场景全局理解的能力，适用于复杂背景的图像分割。

7. unet.py
U-Net
经典的语义分割网络，采用 编码器-解码器结构 与跳跃连接（skip connection），在医学图像和遥感图像中广泛应用，擅长小样本高精度分割。

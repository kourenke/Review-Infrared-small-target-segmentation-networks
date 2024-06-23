# Survey-Infrared-small-target-segmentation-networks

Renke Kou, Chunping Wang, Zhenming Peng, Zhihe Zhao, Yaohong Chen, Jinhui Han, Fuyu Huang, Ying Yu, Qiang Fu,
Infrared small target segmentation networks: A survey,
Pattern Recognition,
2023,
109788,
ISSN 0031-3203,
https://doi.org/10.1016/j.patcog.2023.109788.
(https://www.sciencedirect.com/science/article/pii/S0031320323004867)

Abstract:Fast and robust small target detection is one of the key technologies in the infrared (IR) search and tracking systems. With the development of deep learning, there are many data-driven IR small target segmentation algorithms, but they have not been extensively surveyed; we believe our proposed survey is the first to systematically survey them. Focusing on IR small target segmentation tasks, we summarized 7 characteristics of IR small targets, 3 feature extraction methods, 8 design strategies, 30 segmentation networks, 8 loss functions, and 13 evaluation indexes. Then, the accuracy, robustness, and computational complexities of 18 segmentation networks on 5 public datasets were compared and analyzed. Finally, we have discussed the existing problems and future trends in the field of IR small target detection. The proposed survey is a valuable reference for both beginners adapting to current trends in IR small target detection and researchers already experienced in this field.

1. We have uploaded the label files of 5 public single frame data sets.

      Data set 1 is derived from "Miss Detection vs. False Alarm: Adversarial Learning for Small Object Segmentation in Infrared Images".

      Note: In the dataset MDvsFA, the original author published 10000 images, of which 22 were damaged. We compared them one by one, with IDs 239, 245, 260, 264, 2543, 2553, 2561, 2808, 2817, 2819, 3503, 3504, 3947, 3949, 3962, 7389, 7395, 8094, 8105, 8112, 8757, and 8772, respectively. The actual number is 9978. Therefore, you need to delete the damaged images from 1000 and rename them from 000000 to 009977 to maintain consistency.

      Data set 2 is derived from "AGPCNet: Attention-Guided Pyramid Context Networks for Infrared Small Target Detection".

      Data set 3 is derived from "Asymmetric Contextual Modulation for Infrared Small Target Detection".

      Data set 4 is derived from "Dense Nested Attention Network for Infrared Small Target Detection".

      Data set 5 is derived from "ISNet: Shape Matters for Infrared Small Target Detection".

3. We have uploaded the inference code of infrared small target segmentation, which can directly load pkl files.

4. We have uploaded a lot of pictures of the comparison experiment.

In addition, we have also compiled a set of evaluation metrics libraries suitable for algorithms in this field, named BinarySOSMetrics.

The relevant code is published on https://github.com/BinarySOS/BinarySOSMetrics.

The main features of BinarySOSMetrics include:

High Efficiency: Multi-threading.

Device Friendly: All metrics support automatic batch accumulation.

Unified API: All metrics provide the same API, Metric.update(labels, preds) complete the accumulation of batches， Metric.get() get metrics。

Unified Computational: We use the same calculation logic and algorithms for the same type of metrics, ensuring consistency between results.

Supports multiple data formats: Supports multiple input data formats, hwc/chw/bchw/bhwc/image path, more details in ./notebook/tutorial.ipynb

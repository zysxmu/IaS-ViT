## Evaluation
- Evaluation by the following command:

```bash
python test_quant.py [--model] [--dataset] [--w_bit] [--a_bit] [--iter]

optional arguments:
--model: Model architecture, the choises can be: 
    [vit_small, vit_base, deit_tiny, deit_small, deit_base, swin_tiny, swin_small,swin_base]
--dataset: Path to ImageNet dataset.
--w_bit: Bit-precision of weights, default=4.
--a_bit: Bit-precision of activation, default=4.
--w_cw: Channel-wise weight quantization.
--iter: Iterations of optimization. a3w3/ a4w4 setting is 1000, a6w6 setting is 200.
```

Example: Quantize *DeiT-S* at W4/A4 precision:

```bash
python test_quant.py --model deit_small --dataset <YOUR_DATA_DIR> --w_bit 4 --a_bit 4 --w_cw
```


## ﻿Acknowledge
```
@inproceedings{li2023repq,
  title={Repq-vit: Scale reparameterization for post-training quantization of vision transformers},
  author={Li, Zhikai and Xiao, Junrui and Yang, Lianwei and Gu, Qingyi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={17227--17236},
  year={2023}
}
```


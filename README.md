# Local self-attention in Transformer for visual question answering
LSAT is based on the TRAR_Base[https://github.com/rentainhe/TRAR-VQA] model when using 8*8 grid image features.
### Training
**Train model on VQA-v2 with default hyperparameters:**
```bash
 python3 run.py --RUN='train' --MODEL='mcan_small' --DATASET='vqa'
```
and the training log will be seved to:
```
results/log/log_run_<VERSION>.txt
```
Args:
- `--DATASET={'vqa', 'clevr'}` to choose the task for training
- `--GPU=str`, e.g. `--GPU='2'` to train model on specific GPU device.
- `--SPLIT={'train', 'train+val', train+val+vg'}`, which combines different training datasets. The default training split is `train`.
- `--MAX_EPOCH=int` to set the total training epoch number.


**Resume Training**

Resume training from specific saved model weights
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='lsat' --RESUME=True --CKPT_V=str --CKPT_E=int
```
- `--CKPT_V=str`: the specific checkpoint version
- `--CKPT_E=int`: the resumed epoch number

**Multi-GPU Training and Gradient Accumulation**
1. Multi-GPU Training:
Add `--GPU='0, 1, 2, 3...'` after the training scripts.
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='lsat' --GPU='0,1,2,3'
```
The batch size on each GPU will be divided into `BATCH_SIZE/GPUs` automatically.
**Online Testing**
All the evaluations on the `test` dataset of VQA-v2 and CLEVR benchmarks can be achieved as follows:
```bash
python3 run.py --RUN='test' --MODEL='lsat' --DATASET='{vqa, clevr}' --CKPT_V=str --CKPT_E=int
```

Result file are saved at:

`results/result_test/result_run_<CKPT_V>_<CKPT_E>.json`

You can upload the obtained result json file to [Eval AI](https://evalai.cloudcv.org/web/challenges) to evaluate the scores.



## Acknowledgements
- [openvqa](https://github.com/MILVLG/openvqa)
- [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)
- [TRAR](https://github.com/rentainhe/TRAR-VQA)


## Citation
if LSAT is helpful for your research or you wish to refer the baseline results published here, we'd really appreciate it if you could cite this paper:
```
@article{shen2022local,
  title={Local self-attention in transformer for visual question answering},
  author={Shen, Xiang and Han, Dezhi and Guo, Zihan and Chen, Chongqing and Hua, Jie and Luo, Gaofeng},
  journal={Applied Intelligence},
  pages={1--18},
  year={2022},
  publisher={Springer}
}
```

# Local self-attention in Transformer for visual question answering
### Training
**Train model on VQA-v2 with default hyperparameters:**
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar'
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
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --RESUME=True --CKPT_V=str --CKPT_E=int
```
- `--CKPT_V=str`: the specific checkpoint version
- `--CKPT_E=int`: the resumed epoch number

**Multi-GPU Training and Gradient Accumulation**
1. Multi-GPU Training:
Add `--GPU='0, 1, 2, 3...'` after the training scripts.
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --GPU='0,1,2,3'
```
The batch size on each GPU will be divided into `BATCH_SIZE/GPUs` automatically.
**Online Testing**
All the evaluations on the `test` dataset of VQA-v2 and CLEVR benchmarks can be achieved as follows:
```bash
python3 run.py --RUN='test' --MODEL='trar' --DATASET='{vqa, clevr}' --CKPT_V=str --CKPT_E=int
```

Result file are saved at:

`results/result_test/result_run_<CKPT_V>_<CKPT_E>.json`

You can upload the obtained result json file to [Eval AI](https://evalai.cloudcv.org/web/challenges) to evaluate the scores.

### Models
Here we provide our pretrained model and log, please see [MODEL.md](MODEL.md)

## Acknowledgements
- [openvqa](https://github.com/MILVLG/openvqa)
- [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)
- [TRAR](https://github.com/rentainhe/TRAR-VQA)


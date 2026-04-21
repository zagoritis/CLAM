#!/bin/bash
set -eo pipefail

module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clam
set -u

cd /home/s4076893/Desktop/CLAM

python - <<'PY'
import torch
from torch.utils.data import DataLoader, Subset

from scalant.config import load_config
from scalant.datasets.epickitchens import EpicKitchens
from scalant.models import QueryPredictor
from scalant.criterion import build_criterion
from scalant.utils import Target
from helper import build_optimizer


def move_batch_to_device(batch, device):
    return {
        key: val.to(device=device, non_blocking=True)
        if val is not None and hasattr(val, "to")
        else val
        for key, val in batch.items()
    }


cfg = load_config("configs/ek100/default.yaml")
cfg.DATA.DATA_ROOT_PATH = "/home/s4076893/Desktop"
cfg.TRAIN.BATCH_SIZE = 2
cfg.VAL.BATCH_SIZE = 2
cfg.TRAIN.NUM_WORKERS = 1
cfg.VAL.NUM_WORKERS = 1
cfg.TRAIN.EPOCHS = 1
cfg.TRAIN.SAVE_MODEL = False
cfg.DTYPE = "float32"

device = torch.device("cuda")
print("using", torch.cuda.get_device_name(0))

dataset = EpicKitchens(cfg, "train")
subset = Subset(dataset, [0, 1])
loader = DataLoader(
    subset,
    batch_size=2,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    persistent_workers=True,
    drop_last=False,
)

batch = next(iter(loader))
print({k: tuple(v.shape) for k, v in batch.items()})

batch = move_batch_to_device(batch, device)

model = QueryPredictor(cfg, dataset.num_classes, dataset).to(device).train()
criterion = build_criterion(cfg, dataset)
optimizer = build_optimizer(model, cfg)

target = Target(
    past_feats=batch.get("past_feats"),
    future_feats=batch.get("future_feats"),
    past_actions=batch.get("past_act"),
    past_verbs=batch.get("past_verb"),
    past_nouns=batch.get("past_noun"),
    future_actions=batch.get("future_act"),
    future_verbs=batch.get("future_verb"),
    future_nouns=batch.get("future_noun"),
)

optimizer.zero_grad(set_to_none=True)
pred = model(batch["past_feats"], target)
loss, loss_dict = criterion(pred, target)
loss.backward()
optimizer.step()

loss_summary = {
    key: value
    for key, value in loss_dict.items()
    if isinstance(value, (float, int))
}
print("one-step smoke test ok")
print("loss", float(loss.item()))
print(loss_summary)
PY

"""Headless training run with class-weighted dice loss.

Replicates the training loop in streamlit_label_prep.py exactly -- same optimizer,
same dirt-rate schedule, same per-epoch patch resampling, same checkpoint naming --
but runs without the Streamlit UI and applies per-class weights to the dice half of
the objective.

Settings are read from an existing run's Config.yaml so this is a controlled
comparison: everything matches the reference run except the dice weighting and the
output paths. Reusing the reference run's patches.csv/val.csv keeps the train/val
split identical, which is what makes the two runs comparable at all.

Usage:
    python Scripts/run_weighted_training.py --config <path to Config.yaml> \
        --data-dir <isolated data dir> --out-dir <output dir> [--weights 0.1,0.7,0.2]
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from adabelief_pytorch import AdaBelief
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "MARS" / "morphology" / "segmentation" / "pytorch_segmentation"))

import utils.dataprocess as dp  # noqa: E402
from net.unet_light_rdn import UNet_Light_RDN  # noqa: E402
from utils.dataset import HDF52D, load_patches  # noqa: E402
from utils.generate import get_dirt_bone_patches, random_patches  # noqa: E402
from utils.losses import DomainEnrichLoss, dice_loss  # noqa: E402
from utils.train import rdn_val  # noqa: E402


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Config.yaml from the reference run")
    ap.add_argument("--data-dir", required=True, help="Directory holding the copied dataset.hdf5 and CSVs")
    ap.add_argument("--out-dir", required=True, help="Where checkpoints go")
    ap.add_argument("--weights", default="0.1,0.7,0.2", help="Per-class dice weights: air,non-bone,bone")
    ap.add_argument("--tb-name", default="weighted_dice", help="Suffix for the TensorBoard run directory")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Override the epoch count from the config (used for smoke tests)")
    ap.add_argument("--limit-patches", type=int, default=None,
                    help="Use only the first N training patches (used for smoke tests)")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_weights = [float(x) for x in args.weights.split(",")]
    log(f"dice class weights (air, non-bone, bone): {class_weights}  sum={sum(class_weights):.4f}")

    # Everything below mirrors the app; only the paths and the weighting differ.
    use_gpu = config["gpu_config"]["use_gpu"]
    gpu_name = config["gpu_config"]["gpu_name"]
    class_num = config["model"]["class_num"]
    batch_size = int(config["data_loader"]["batch_size"])
    workers = int(config["data_loader"].get("num_workers", 0))
    output_size = config["output_size"]
    epochs = int(config["train_param"]["Epoch"])
    if args.max_epochs is not None:
        epochs = args.max_epochs
        log(f"epoch count overridden to {epochs}")

    period = config.get("period")
    if period is None:
        period = 8  # same fallback the app uses

    if use_gpu:
        torch.cuda.set_device(gpu_name)

    net = UNet_Light_RDN(n_channels=config["model"]["n_channels"], n_classes=class_num)
    pretrained = config["model"]["path"]
    if str(config["model"]["if_pre_train"]).lower() == "true" and pretrained:
        map_loc = torch.device(type="cuda", index=gpu_name) if use_gpu else torch.device("cpu")
        net.load_state_dict(torch.load(pretrained, map_location=map_loc, weights_only=True))
        log(f"loaded pretrained weights from {pretrained}")
    else:
        log("training from scratch (no pretrained weights)")

    if str(config["optimizer"]["method"]) == "AdaBelief":
        optimizer = AdaBelief(
            net.parameters(),
            lr=float(config["optimizer"]["lr"]),
            eps=float(config["optimizer"]["epsilon"]),
            betas=(0.9, 0.999),
            weight_decouple=True,
            rectify=False,
        )
    else:
        optimizer = getattr(torch.optim, config["optimizer"]["method"])(
            net.parameters(),
            lr=float(config["optimizer"]["lr"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
    log(f"optimizer {config['optimizer']['method']} lr={config['optimizer']['lr']} "
        f"batch={batch_size} epochs={epochs} period={period}")

    train_patches = load_patches(str(data_dir / "patches.csv"))
    val_patches = load_patches(str(data_dir / "val.csv"))
    ratios = load_patches(str(data_dir / "ratios.csv"))
    h5 = str(data_dir / "dataset.hdf5")
    if args.limit_patches is not None:
        train_patches = train_patches[:args.limit_patches]
        ratios = ratios[:args.limit_patches]
        log(f"patches limited to {len(train_patches)} for smoke test")
    log(f"{len(train_patches)} train patches, {len(val_patches)} val patches from {data_dir}")

    train_transform = transforms.Compose([
        dp.Augmentation(output_size=output_size),
        dp.AdjustMask(class_num=class_num),
        dp.Normalize(input_max=255, input_min=0),
        dp.ToTensor(),
    ])
    val_transform = transforms.Compose([
        dp.AdjustMask(class_num=class_num),
        dp.Normalize(input_max=255, input_min=0),
        dp.ToTensor(),
    ])

    tb_dir = REPO / "runs" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.tb_name}"
    writer = SummaryWriter(str(tb_dir))
    log(f"TensorBoard: {tb_dir}")

    # Record exactly what this run used, next to its checkpoints.
    run_config = dict(config)
    run_config["path"] = {"data_path": h5, "save_path": str(out_dir)}
    run_config["csv_path"] = {
        "train": str(data_dir / "patches.csv"),
        "val": str(data_dir / "val.csv"),
        "ratios": str(data_dir / "ratios.csv"),
    }
    run_config["dice_class_weights"] = class_weights
    with open(out_dir / "Config.yaml", "w") as f:
        yaml.dump(run_config, f)

    bce_losses = nn.BCEWithLogitsLoss()
    global_step = 0
    total_timer = time.time()

    for i_epoch in range(epochs):
        # Same non-bone ramp the app uses.
        if i_epoch < period:
            dirt_rate = 0.5
        elif i_epoch < 2 * period:
            dirt_rate = 0.3
        elif i_epoch < 3 * period:
            dirt_rate = 0.1
        else:
            dirt_rate = 0.0

        new_patches = random_patches(dirt_choose_threshold=0.1, dirt_rate=dirt_rate,
                                     patches=train_patches, ratios=ratios)
        rdn_patches, index = get_dirt_bone_patches(train_patches, ratios)

        data_set1 = HDF52D(h5, new_patches, val_patches,
                           train_transform=train_transform, val_transform=val_transform)
        data_set2 = HDF52D(h5, rdn_patches, val_patches,
                           train_transform=train_transform, val_transform=val_transform,
                           train_idx=index)

        loader1 = DataLoader(data_set1, batch_size=batch_size, shuffle=True,
                             pin_memory=True, num_workers=workers)
        loader2 = DataLoader(data_set2, batch_size=batch_size, shuffle=True,
                             pin_memory=True, num_workers=workers)

        net.cuda() if use_gpu else net.cpu()
        net.train()

        it2 = iter(loader2)
        loss1_sum = loss2_sum = 0.0
        n_batches = 0
        epoch_timer = time.time()

        for batch in loader1:
            try:
                b2 = next(it2)
            except StopIteration:
                it2 = iter(loader2)
                b2 = next(it2)

            mask = batch["mask"]
            image = batch["image"]
            image2 = b2["image"]
            idx = b2["index"]
            if use_gpu:
                mask = mask.cuda().long()
                image = image.cuda()
                image2 = image2.cuda()

            net(image2)
            loss1 = DomainEnrichLoss()(net, idx)

            logits = net(image)
            onehot = dp.create_one_hot(mask)
            probs = torch.sigmoid(logits)
            loss2 = (0.25 * bce_losses(logits, onehot)
                     + 0.75 * dice_loss(probs, onehot, class_weights=class_weights))
            loss = loss2 + loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss1_sum += float(loss1.detach().cpu())
            loss2_sum += float(loss2.detach().cpu())
            n_batches += 1
            writer.add_scalars("Losses", {
                "total": float(loss.detach().cpu()),
                "loss1_domain_enrich": float(loss1.detach().cpu()),
                "loss2_bce_weighted_dice": float(loss2.detach().cpu()),
            }, global_step)
            global_step += 1

        writer.add_scalars("Average_losses", {
            "loss1_domain_enrich": loss1_sum / max(n_batches, 1),
            "loss2_bce_weighted_dice": loss2_sum / max(n_batches, 1),
        }, global_step)

        val_loss, class_val = rdn_val(net, data_set1, use_gpu=use_gpu,
                                      i_epoch=i_epoch, class_num=class_num)
        names = ["air", "non_bone", "bone"]
        writer.add_scalars("Dice_overlap",
                           {names[i]: float(class_val[i]) for i in range(len(class_val))},
                           i_epoch)
        writer.add_scalar("val_accuracy", float(val_loss), i_epoch)
        writer.flush()

        save_name = out_dir / f"Loss-{i_epoch}_{val_loss:.6f}.pth"
        torch.save(net.state_dict(), save_name)
        log(f"epoch {i_epoch + 1}/{epochs} done in {time.time() - epoch_timer:.0f}s "
            f"| dirt_rate={dirt_rate} val_acc={val_loss:.6f} "
            f"| dice air={class_val[0]:.4f} non-bone={class_val[1]:.4f} bone={class_val[2]:.4f} "
            f"| saved {save_name.name}")

    writer.close()
    log(f"TRAINING COMPLETE in {(time.time() - total_timer) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()

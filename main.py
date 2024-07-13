import numpy as np
import torch

import hydra
import wandb
from torchvision import transforms
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


from src.util import *
from src.dataset import *
from src.model import *
from src.criterion import *

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    set_seed(cfg.seed)
    transform = transforms.Compose([
        transforms.Resize(size=235, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
    ])

    wandb_logger = WandbLogger(project="DL_Matsuo")

    train_dataset = NewVQA(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = NewVQA(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    num_label = len(list(train_dataset.label_encoder.classes_))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=18)
    valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=18)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    trainer = L.Trainer(accelerator='auto', devices=cfg.devices, max_epochs=cfg.epochs, log_every_n_steps=1,
                        enable_checkpointing=True, default_root_dir="./saved/model/",
                        enable_progress_bar=True, enable_model_summary=True, accumulate_grad_batches=1,
                        logger=wandb_logger,
                        callbacks=[EarlyStopping(monitor="val_total_acc", min_delta=0.00, patience=3, verbose=False, mode="max")],
              )

    if cfg.infer_only:
        model = LitVQA.load_from_checkpoint(
            '/home/acg16548yb/dl_lecture_competition_pub/DL_Matsuo/8xiha1qt/checkpoints/epoch=43-step=6864.ckpt',
            cfg=cfg,
            label=num_label,
        )
    else:
        model = LitVQA(cfg, label=num_label)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


    submission = trainer.predict(model=model, dataloaders=test_loader)
    submission = train_dataset.label_encoder.inverse_transform(submission)
    submission = np.array(submission)
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()

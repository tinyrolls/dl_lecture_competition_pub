import numpy as np
import torch
import timm
from torch import nn
import lightning as L
from src.criterion import *
import torch.nn.functional as F
from torch_geometric.nn import Sequential
from transformers import AutoModel

class LitVQA(L.LightningModule):
    def __init__(self, cfg, label):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.vqa = VQAModel(cfg, label)

    def forward(self, image, question):
        x = self.vqa(image, question)
        return x

    def training_step(self, batch, batch_idx):
        image, question, answers, mode_answer = batch
        pred = self.vqa(image, question)
        loss = F.cross_entropy(pred, mode_answer.squeeze())
        self.log("train_loss", loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, logger=True)

        total_acc = VQA_criterion(pred.argmax(1), answers)
        simple_acc = (pred.argmax(1) == mode_answer).float().mean().item()
        self.log("total_acc", total_acc, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, logger=True)
        self.log("simple_acc", simple_acc, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        image, question, answers, mode_answer = batch
        pred = self.vqa(image, question)
        loss = F.cross_entropy(pred, mode_answer.squeeze())
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True, logger=True)

        total_acc = VQA_criterion(pred.argmax(1), answers)
        simple_acc = (pred.argmax(1) == mode_answer).float().mean().item()
        self.log("val_total_acc", total_acc, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True, logger=True)
        self.log("val_simple_acc", simple_acc, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        return optimizer

    def predict_step(self, batch):
        image, question = batch
        pred = self.vqa(image, question)
        pred = pred.argmax(1).cpu().item()
        return pred

class VQAModel(nn.Module):
    def __init__(self, cfg, label):
        super().__init__()
        self.cfg = cfg

        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")

        # Freeze all layers in BERT except the last two layers
        for name, param in self.text_encoder.named_parameters():
            if not name.startswith('encoder.layer.11') and not name.startswith('encoder.layer.10'):
                param.requires_grad = False

        self.resnet = Sequential('x', [
            (timm.create_model(
                cfg.image_model_name,
                pretrained=True,
                num_classes=0,  # remove classifier nn.Linear
            ), 'x -> x'),
            nn.Linear(cfg.image_out_dim, cfg.image_emb_dim),
        ])

        self.fc = nn.Sequential(
            nn.Linear(cfg.image_emb_dim+cfg.text_emb_dim, cfg.fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.fc_hidden_dim, label)
        )

    def forward(self, image, question):
        image_feature = self.resnet(image)
        model_output = self.text_encoder(**question)
        sentence_embeddings = mean_pooling(model_output, question['attention_mask'])
        question_feature = F.normalize(sentence_embeddings, p=2, dim=1)

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

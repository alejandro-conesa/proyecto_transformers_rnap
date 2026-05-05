import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
import pandas as pd
import kagglehub
import os
import re
from tokenizers import BertWordPieceTokenizer
from sklearn.model_selection import train_test_split

def limpiar(texto):
    texto = re.sub(r"<.*?>", "", texto)      # quitar HTML
    texto = re.sub(r"[^a-zA-Z ]", "", texto) # solo letras
    return texto.lower().strip()

def separar_dataset(df):
    df["review_clean"] = df["review"].apply(limpiar)

    X = df["review_clean"]
    y = df["sentiment"].map({"positive": 1, "negative": 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def crear_vocabulario(X_train, tokenizer_folder_path='./mi_tokenizer'):
    with open("corpus.txt", "w") as f:
        f.write("\n".join(X_train))

    tokenizer = BertWordPieceTokenizer(lowercase=True)
    tokenizer.train(
        files=["corpus.txt"],
        vocab_size=10000,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    os.makedirs(tokenizer_folder_path, exist_ok=True)
    tokenizer.save_model(tokenizer_folder_path)

class IMDBDataset(Dataset):
    def __init__(self, textos, etiquetas, tokenizer, max_len=256):
        self.textos = textos
        self.etiquetas = etiquetas
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.textos[idx])

        ids = encoded.ids[:self.max_len]
        mask = [1] * len(ids)

        # Padding
        pad_len = self.max_len - len(ids)
        ids  += [0] * pad_len
        mask += [0] * pad_len

        return {
            "input_ids":      torch.tensor(ids,  dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label":          torch.tensor(self.etiquetas[idx], dtype=torch.long)
        }

class IMDBDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32, max_len=256, tokenizer_path='./mi_tokenizer'):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer_path = tokenizer_path

    def prepare_data(self):
        path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        df = pd.read_csv(f"{path}/IMDB Dataset.csv")
        df["review_clean"] = df["review"].apply(limpiar)

        X = df["review_clean"]
        y = df["sentiment"].map({"positive": 1, "negative": 0})

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        if not os.path.exists(f"{self.tokenizer_path}/vocab.txt"):
            crear_vocabulario(X_train, self.tokenizer_path)

    def setup(self, stage=None):
        path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        df = pd.read_csv(f"{path}/IMDB Dataset.csv")
        df["review_clean"] = df["review"].apply(limpiar)

        X = df["review_clean"]
        y = df["sentiment"].map({"positive": 1, "negative": 0})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        tokenizer = BertWordPieceTokenizer(f"{self.tokenizer_path}/vocab.txt", lowercase=True)

        if stage == "fit" or stage is None:
            self.train_dataset = IMDBDataset(X_train.values, y_train.values, tokenizer, self.max_len)
            self.val_dataset   = IMDBDataset(X_val.values,   y_val.values,   tokenizer, self.max_len)

        if stage == "test" or stage is None:
            self.test_dataset  = IMDBDataset(X_test.values,  y_test.values,  tokenizer, self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
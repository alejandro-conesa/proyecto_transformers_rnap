from datos import IMDBDataModule
from modelo import IMDBModel
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary

EPOCHS = 10
NAME   = "mini_bert"


dm    = IMDBDataModule(batch_size=64, max_len=256)
model = IMDBModel(vocab_size=10_000, num_layers=4, hidden_dim=256)

early_stopping = EarlyStopping(monitor="val/loss", patience=3, mode="min")
ckpt = ModelCheckpoint(
    dirpath="weights",
    filename=f"{EPOCHS}ep-{NAME}",
    verbose=True,
    monitor="val/loss", 
    mode="min"
)
summary = ModelSummary(max_depth=3)

trainer = L.Trainer(max_epochs=10, accelerator="auto", callbacks=[early_stopping, ckpt, summary])
trainer.fit(model, dm)
trainer.test(model, dm)
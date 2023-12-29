from pathlib import Path

import fastai.tabular.all as fai_tab
import pandas as pd
import torch

pd.options.display.max_columns = 0


class DeepLearningEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass


if __name__ == "__main__":
    comp = "titanic"
    path = Path(f"./datasets/{comp}")
    df = pd.read_csv(path / "train.csv")
    splits = fai_tab.RandomSplitter(seed=42)(df)
    dls = fai_tab.TabularPandas(
        df,
        splits=splits,
        procs=[fai_tab.Categorify, fai_tab.FillMissing, fai_tab.Normalize],
        cat_names=["Sex", "Pclass", "Embarked"],
        cont_names=["Age", "SibSp", "Parch", "Fare"],
        y_names="Survived",
        y_block=fai_tab.CategoryBlock(),
    ).dataloaders(path=".")

    model_emb = DeepLearningEmbeddings()
    learn = fai_tab.TabularLearner(dls, model_emb, loss_func=fai_tab.BCELossFlat())
    learn.fit(5, lr=0.03)
    print("here")

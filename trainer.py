from nocola_to_df import get_df
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from models import SBERT_MODEL
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from constants import label_mapping

# get df for all splits:
train_df = get_df(split="train")
dev_df = get_df(split="dev")
test_df = get_df(split="test")

train_df["label"] = train_df["label"].map(label_mapping)
test_df["label"] = test_df["label"].map(label_mapping)
dev_df["label"] = dev_df["label"].map(label_mapping)

def get_reduced_dataframe(df, min_samples=8):
    # proportion of each label, based on the minimum label
    proportion = df.label.value_counts() / min_samples
    # normalize it, so the lowest value is 1:
    proportion = proportion  / proportion.min()
    # multiply each by 8 (so that we ensure at least 8 samples)
    proportion = proportion * min_samples
    proportion = proportion.apply(round)
    proportion = proportion.to_dict()
    
    subset_dfs = []
    for label, desired_count in proportion.items():
        label_df = df[df['label'] == label]
        subset_dfs.append(label_df.sample(n=desired_count, random_state=42))
    subset = pd.concat(subset_dfs, axis=0).reset_index(drop=True)
    return subset

subset_train = get_reduced_dataframe(train_df, min_samples=20)

dataset = Dataset.from_pandas(subset_train)
dataset_test = Dataset.from_pandas(test_df)
dataset_eval = Dataset.from_pandas(dev_df)

num_classes = len(dataset.unique("label"))

sfit_model = SetFitModel.from_pretrained(
    SBERT_MODEL,
    use_differentiable_head=True,
    head_params={"out_features": num_classes},
)
#Note: If you use the differentiable SetFitHead classifier head, it will automatically use BCEWithLogitsLoss for training.
# The prediction involves a sigmoid after which probabilities are rounded to 1 or 0.
# Furthermore, the "one-vs-rest" and "multi-output" multi-target strategies are equivalent for the differentiable SetFitHead.
# Create trainer
trainer = SetFitTrainer(
    model=sfit_model,
    train_dataset=dataset,
    eval_dataset=dataset_eval,
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=16,
    num_iterations=20, # Number of text pairs to generate for contrastive learning
    num_epochs=1, # Number of epochs to use for contrastive learning
)
trainer.freeze()
trainer.train()
trainer.unfreeze(keep_body_frozen=True)
trainer.train(
    num_epochs=25,
    batch_size=16,
    body_learning_rate=1e-5,  # LR of body
    learning_rate=1e-2,  # LR of head
    l2_weight=0.0
)
metrics = trainer.evaluate()
print(metrics)
trainer.push_to_hub("setfit-nocola-20-iter-25-epochs")
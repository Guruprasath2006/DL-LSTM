 # DL- Developing a Deep Learning Model for NER using LSTM



## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
The objective of this project is to build, train, and evaluate a Bidirectional Long Short-Term Memory (BiLSTM) deep neural network using PyTorch to perform Named Entity Recognition (NER). Given an input sequence of tokens, the network identifies and assigns sequence tags corresponding to specific entity categories (such as Person, Organization, Geopolitical Entity, Time, etc.) or tags tokens as non-entities (O).

The experiment uses the ner_dataset.csv corpus (typically derived from the Kaggle/GMB annotated NER dataset). It contains sentences structured across four primary columns: Sentence #, Word, POS (Part-of-Speech), and Tag (IOB-formatted named entity tags such as B-geo, I-org, B-per, O, etc.).

## DESIGN STEPS
### STEP 1: 

Load ner_dataset.csv using Pandas and handle missing sentence indices using forward-fill (ffill). Extract unique tokens and tags to build index-to-token/tag and token/tag-to-index mapping dictionaries (word2idx, tag2idx, idx2tag). Add an explicit ENDPAD token to represent padding elements.

### STEP 2: 
Group the tokens and their matching entity tags by sentence using a helper class (SentenceGetter). Convert word and tag tokens into numerical sequences using index mappings. Inspect sequence lengths via a histogram and pad/truncate all sequences to a fixed length (max_len = 50) using PyTorch’s pad_sequence, setting pad values to ENDPAD for words and O for labels.


### STEP 3: 

Split the padded input tensors (X_pad, y_pad) into training and testing partitions using an 80:20 split (train_test_split). Wrap the arrays into a custom PyTorch Dataset (NERDataset) and pass them into DataLoader objects with a batch size of 32 for batching and shuffling.

### STEP 4: 

Construct a sequence-tagging module (BiLSTMTagger) inheriting from nn.Module

### STEP 5: 

Instantiate the model and place it on the GPU/CPU device. Define the loss criterion using nn.CrossEntropyLoss and the optimizer as torch.optim.Adam (learning rate = 0.001). Train across multiple epochs by flattening predictions and targets to compute cross-entropy, executing backpropagation (loss.backward()), and recording training and validation loss trajectories.

### STEP 6: 


Evaluate model performance across the test set by computing a token-level classification_report (Precision, Recall, F1-Score) while filtering out ENDPAD tokens. Plot the training vs. validation loss curve across epochs, and perform inference on a sample test sentence comparing the ground-truth tags against predicted entity tags.




## PROGRAM



```python
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




data = pd.read_csv("/content/ner_dataset.csv", encoding="latin1").ffill()

words = list(data["Word"].unique())
tags = list(data["Tag"].unique())

if "ENDPAD" not in words:
    words.append("ENDPAD")

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}

data.head(50)


# Entity descriptions
entity_types = {
    "geo": "Geographical Entity",
    "org": "Organization",
    "per": "Person",
    "gpe": "Geopolitical Entity",
    "tim": "Time indicator",
    "art": "Artifact",
    "eve": "Event",
    "nat": "Natural Phenomenon"
}

print("Unique words in corpus:", data["Word"].nunique())
print("Unique tags in corpus:", data["Tag"].nunique())

print("Unique tags are:", tags)




class SentenceGetter:
    def __init__(self, data):
        self.grouped = data.groupby(
            "Sentence #",
            group_keys=False
        ).apply(
            lambda s: [
                (w, t)
                for w, t in zip(s["Word"], s["Tag"])
            ]
        )

        self.sentences = list(self.grouped)


getter = SentenceGetter(data)
sentences = getter.sentences

print(sentences[35])




X = [
    [word2idx[w] for w, t in sentence]
    for sentence in sentences
]

y = [
    [tag2idx[t] for w, t in sentence]
    for sentence in sentences
]

print(word2idx)




plt.hist([len(s) for s in sentences], bins=50)
plt.title("Sentence Length Distribution")
plt.xlabel("Sentence Length")
plt.ylabel("Frequency")
plt.show()



max_len = 50

X_pad = pad_sequence(
    [torch.tensor(seq, dtype=torch.long) for seq in X],
    batch_first=True,
    padding_value=word2idx["ENDPAD"]
)

y_pad = pad_sequence(
    [torch.tensor(seq, dtype=torch.long) for seq in y],
    batch_first=True,
    padding_value=tag2idx["O"]
)

X_pad = X_pad[:, :max_len]
y_pad = y_pad[:, :max_len]

print(X_pad[0])
print(y_pad[0])




X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    y_pad,
    test_size=0.2,
    random_state=1
)




class NERDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "input_ids": self.X[idx],
            "labels": self.y[idx]
        }


train_loader = DataLoader(
    NERDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    NERDataset(X_test, y_test),
    batch_size=32
)




class BiLSTMTagger(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_tags,
        padding_idx
    ):
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            num_tags
        )

    def forward(self, input_ids):

        embeddings = self.embedding(input_ids)

        lstm_out, _ = self.lstm(embeddings)

        output = self.fc(lstm_out)

        return output




VOCAB_SIZE = len(word2idx) + 1
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_TAGS = len(tag2idx)

model = BiLSTMTagger(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_tags=NUM_TAGS,
    padding_idx=word2idx["ENDPAD"]
).to(device)


loss_fn = nn.CrossEntropyLoss(
    ignore_index=tag2idx["O"]
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)




def train_model(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    epochs=3
):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        

        model.train()
        total_train_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)

            loss = loss_fn(
                outputs.view(-1, outputs.shape[-1]),
                labels.view(-1)
            )

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        train_losses.append(avg_train_loss)




        model.eval()
        total_val_loss = 0

        with torch.no_grad():

            for batch in test_loader:

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)

                loss = loss_fn(
                    outputs.view(-1, outputs.shape[-1]),
                    labels.view(-1)
                )

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)

        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

    return train_losses, val_losses




def evaluate_model(model, test_loader):

    model.eval()

    true_tags = []
    pred_tags = []

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)

            preds = torch.argmax(
                outputs,
                dim=-1
            )

            for i in range(len(labels)):

                for j in range(len(labels[i])):

                    
                    if (
                        input_ids[i][j].item()
                        != word2idx["ENDPAD"]
                    ):

                        true_tags.append(
                            idx2tag[
                                labels[i][j].item()
                            ]
                        )

                        pred_tags.append(
                            idx2tag[
                                preds[i][j].item()
                            ]
                        )

    print("\nClassification Report:\n")

    print(
        classification_report(
            true_tags,
            pred_tags,
            zero_division=0
        )
    )




train_losses, val_losses = train_model(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    epochs=3
)

evaluate_model(
    model,
    test_loader
)




print("Name: ")
print("Register Number: ")

history_df = pd.DataFrame({
    "loss": train_losses,
    "val_loss": val_losses
})

history_df.plot(
    title="Loss Over Epochs"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()




i = 125

model.eval()

sample = X_test[i].unsqueeze(0).to(device)

with torch.no_grad():

    output = model(sample)

    preds = torch.argmax(
        output,
        dim=-1
    ).squeeze().cpu().numpy()


true = y_test[i].numpy()



print(
    "{:<15} {:<10} {}\n{}".format(
        "Word",
        "True",
        "Pred",
        "-" * 40
    )
)


for w_id, true_tag, pred_tag in zip(
    X_test[i],
    y_test[i],
    preds
):

    if w_id.item() != word2idx["ENDPAD"]:

        word = words[w_id.item() - 1]

        true_label = idx2tag[
            true_tag.item()
        ]

        pred_label = idx2tag[
            int(pred_tag)
        ]

        print(
            f"{word:<15} "
            f"{true_label:<10} "
            f"{pred_label}"
        )


```

### OUTPUT

## Loss Vs Epoch Plot

<img width="705" height="565" alt="image" src="https://github.com/user-attachments/assets/302b46db-880a-465b-9bf2-774be907db26" />


### Sample Text Prediction
<img width="431" height="427" alt="image" src="https://github.com/user-attachments/assets/c7cb7c87-18da-43cb-b84e-c094d3229cd3" />


## RESULT
Thus, a Bidirectional LSTM (BiLSTM) sequence labeling model for Named Entity Recognition was successfully executed.

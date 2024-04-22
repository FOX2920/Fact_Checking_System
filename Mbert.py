import torch
from torch import nn
from torch.utils.data import Dataset

class SentencePairDataset(Dataset):
  def __init__(self, sentence_pairs, labels, tokenizer, max_length):
    self.sentence_pairs = sentence_pairs
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.sentence_pairs)

  def __getitem__(self, idx):
    sentence1, sentence2 = self.sentence_pairs[idx]
    label = self.labels[idx]
    encoding = self.tokenizer.encode_plus(
        sentence1,
        text_pair=sentence2,
        add_special_tokens=True,
        max_length=self.max_length,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )
    return {
        "input_ids": encoding["input_ids"].flatten(),
        "attention_mask": encoding["attention_mask"].flatten(),
        "label": torch.tensor(label, dtype=torch.long),
    }
  
class MBERTClassifier(nn.Module):
    def __init__(self, mbert, num_classes):
        super(MBERTClassifier, self).__init__()
        self.mbert = mbert
        self.layer_norm = nn.LayerNorm(self.mbert.config.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(self.mbert.config.hidden_size)
        self.linear = nn.LazyLinear(num_classes)
        self.activation = nn.ELU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.mbert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        norm_output = self.layer_norm(pooled_output)
        batch_norm_output = self.batch_norm(norm_output)
        logits = self.linear(batch_norm_output)
        activated_output = self.activation(logits)
        dropout_output = self.dropout(activated_output)
        return dropout_output

    def predict_proba(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities
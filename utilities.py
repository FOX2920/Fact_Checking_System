import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)

import torch
import numpy as np
import random
from transformers import AutoModel, AutoTokenizer
from Mbert import MBERTClassifier, SentencePairDataset
from torch.utils.data import DataLoader
from evidence_retrieval import evidence_top_n, similarities
from config import hf_token
import pandas as pd

# Thiết lập seed cố định
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Gọi hàm set_seed với seed cố định, ví dụ: 42
set_seed(42)
device = torch.device("cpu")
modelname = "SonFox2920/MBert_FC"
tokenizer = AutoTokenizer.from_pretrained(modelname, token=hf_token)
mbert = AutoModel.from_pretrained(modelname, token=hf_token).to(device)
model = MBERTClassifier(mbert, num_classes=3).to(device)
model.load_state_dict(torch.load('Model/checkpoint.pt', map_location=device))

def predict(context, claim):
    data = pd.DataFrame([{'context': context, 'claim': claim}])

    list_evidence_top5 = []
    list_evidence_top1 = []

    for i in range(len(data)):
        statement = data.claim[i]
        context = data.context[i]
        evidence_top5, top5_consine = evidence_top_n(context, statement)
        evidence_top1, top1_consine, rank_5 = similarities(evidence_top5, statement, top5_consine)
        evidence_top1 = "".join(evidence_top1)
        list_evidence_top5.append(rank_5)
        list_evidence_top1.append(evidence_top1)

    data['evidence_top5'] = list_evidence_top5
    data['evidence'] = list_evidence_top1

    X1_pub_test = data['claim']
    X2_pub_test = data['context']
    X_pub_test = [(X1_pub_test, X2_pub_test) for (X1_pub_test, X2_pub_test) in zip(X1_pub_test, X2_pub_test)]
    y_pub_test = [1]

    test_dataset = SentencePairDataset(X_pub_test, y_pub_test, tokenizer, 256)
    test_loader_pub = DataLoader(test_dataset, batch_size=1)

    model.eval()
    predictions = []
    probabilities = []

    for batch in test_loader_pub:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)
            predictions.extend(predicted.cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())

    data['verdict'] = predictions
    data['verdict'] = data['verdict'].replace(0, "SUPPORTED")
    data['verdict'] = data['verdict'].replace(1, "REFUTED")
    data['verdict'] = data['verdict'].replace(2, "NEI")

    result = {
        'verdict': data['verdict'][0],
        'evidence': data['evidence'][0],
        'probabilities': {
            'SUPPORTED': probabilities[0][0],
            'REFUTED': probabilities[0][1],
            'NEI': probabilities[0][2]
        }
    }
    
    return result

# # Set default context and claim
# context = "Trái Đất là hành tinh duy nhất trong Hệ Mặt Trời được biết đến là nơi có sự sống tồn tại. Nó là hành tinh lớn thứ ba trong hệ này về kích thước và khối lượng. Trái Đất hình cầu với bề mặt gồm nước và đất liền, được bao phủ bởi lớp khí quyển. Khí quyển của Trái Đất chủ yếu bao gồm nitơ và oxy, cùng với các khí nhà kính như hơi nước và carbon dioxide. Trái Đất quay quanh Mặt Trời theo một quỹ đạo hình ellip, hoàn thành một vòng quay trong khoảng 365 ngày, gây ra sự luân phiên của các mùa."
# claim = "Trái Đất là hành tinh duy nhất trong Hệ Mặt Trời được biết đến là nơi có sự sống tồn tại."

# verdict, probabilities = predict(context, claim)

# print(f"Verdict: {verdict}")

# # Display percentages with colors
# labels = ["SUPPORTED", "REFUTED", "NEI"]
# for i, label in enumerate(labels):
#     print(f'{label}: {probabilities[0][i]*100:.2f}%')

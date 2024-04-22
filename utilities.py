import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)

import torch
from transformers import AutoModel, AutoTokenizer
from Mbert import MBERTClassifier
from evidence_retrieval import evidence_top_n, similarities
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelname = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(modelname)
mbert = AutoModel.from_pretrained(modelname).to(device)
model = MBERTClassifier(mbert, num_classes=3).to(device)
model.load_state_dict(torch.load('Model/checkpoint.pt'))

# Hàm để dự đoán nhãn
def predict(context, claim):

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    evidence_top5, top5_consine = evidence_top_n(context, claim)
    evidence_top1, top1_consine, rank_5 = similarities(evidence_top5, claim, top5_consine)

    inputs = tokenizer(context, claim, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        logits = outputs
        probabilities = torch.softmax(logits, dim=-1)
        predicted_label_index = torch.argmax(logits, dim=1)
        label = ["SUPPORTED", "REFUTED", "NEI"]
        predicted_label = label[predicted_label_index.item()]

    # Làm tròn phần trăm đến 2 chữ số thập phân
    probabilities_rounded = [round(float(prob), 2) for prob in probabilities[0]]

    # Tạo dictionary chứa kết quả
    result = {
        "predicted_label": predicted_label,
        "evidence": evidence_top1,
        "probabilities": {
            "SUPPORTED": probabilities_rounded[0],
            "REFUTED": probabilities_rounded[1],
            "NEI": probabilities_rounded[2]
        }
    }

    # Convert dictionary thành JSON và trả về
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
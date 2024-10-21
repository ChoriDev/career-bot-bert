import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 모델 파일 경로 설정
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model/career_maturity_classifier.pth')

class Evaluator:
    def __init__(self):
        target_model = 'klue/bert-base'
        num_labels = 3  # 상, 중, 하를 3개의 클래스로 설정
        self.tokenizer = BertTokenizer.from_pretrained(target_model)
        self.model = BertForSequenceClassification.from_pretrained(target_model, num_labels=num_labels)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
    
    def predict(self, sentence):
        max_len = 512

        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, max_length=max_len, truncation=True)
        inputs = {key: value for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1)

        return prediction.item() + 1
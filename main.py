import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from model import Item
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 모델 파일 경로 설정
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model/career_maturity_classifier.pth')

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

target_model = 'klue/bert-base'
num_labels = 3  # 상, 중, 하를 3개의 클래스로 설정
tokenizer = BertTokenizer.from_pretrained(target_model)
model = BertForSequenceClassification.from_pretrained(target_model, num_labels=num_labels)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # 모델을 평가 모드로 설정

@app.post('/grade')
async def receive_data(item: Item):
    inputs = tokenizer(item.sentence, return_tensors="pt", padding=True, max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)

    return {'grade': prediction.item() + 1}
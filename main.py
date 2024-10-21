from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from model import Item
from evaluator import Evaluator

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/grade')
async def receive_data(item: Item):
    evaluator = Evaluator()
    grade = evaluator.predict(item.sentence)
    return {'grade': grade}
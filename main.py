from fastapi import FastAPI, HTTPException
from transformers import pipeline

from app.models import Message


app = FastAPI()


summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")


@app.post("/summarize")
async def summarize(message: Message):

    if not message.text:
        raise HTTPException(status_code=400, detail="Enter some text, your message cannot be empty")

    try:
        summary = summarization_pipeline(message.text, max_length=150, min_length=40, do_sample=False)
        summary_text = summary[0]['summary_text']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during summarization: {e}")

    return {"summary": summary_text}

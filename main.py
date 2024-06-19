from fastapi import FastAPI, HTTPException
from transformers import pipeline

from app.models import Message


app = FastAPI()


summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")


@app.post("/summarize")
async def summarize(message: Message):

    """
    Summarize text using a summarization pipeline.

    Parameters:
    - message (Message): The input message containing the text to summarize.

    Returns:
    - json: A json containing the summarized text.

    Raises:
    - HTTPException: If the input text is empty or if there's an error during summarization.
    """

    if not message.text:
        raise HTTPException(status_code=400, detail="Enter some text, your message cannot be empty")

    try:
        summary = summarization_pipeline(message.text, max_length=150, min_length=40, do_sample=False)
        summary_text = summary[0]["summary_text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during summarization: {e}")

    return {"summary": summary_text}

# Text-summarizer

This is FastAPI service, which makes summary for input text.

# Used technologies
- Programming language: Python
- Frameworks: FastAPI
- Model source: LangChain

# How to use
To clone this project from GitHub, follow these steps:

Open your terminal or command prompt.
```shell
git clone git@github.com:NataliiaKorzh/text-summarizer.git

```
Run the following commands:
```shell
python -m venv venv
venv\Scripts\activate (on Windows)
source venv/bin/activate (on macOS)
pip install -r requirements.txt

```

# Run server

```shell
uvicorn main:app --reload
```
For testing summarize endpoint open a browser and enter url

http://127.0.0.1:8000/docs/

- POST request

```
{
  "text": "string"
}
```
- Response

```
{
  "summary": "string"
}
```

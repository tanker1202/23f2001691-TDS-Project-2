import os
import io
import httpx
import duckdb
import pytesseract
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

OPENAI_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDE2OTFAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.bytu_vnkiC5Fkn0lhzLRzgCjMRSBxOU5rwOoVxT6hzs"
OPENAI_API_URL = "https://aipipe.org/openrouter/v1/chat/completions"


# ---------- File Extractors ----------

def extract_text_from_image(file: UploadFile) -> str:
    """OCR extract text from image."""
    image = Image.open(io.BytesIO(file.file.read()))
    text = pytesseract.image_to_string(image)
    file.file.seek(0)
    return text


def load_dataframe(file: UploadFile) -> pd.DataFrame:
    """Load CSV, Excel, or Parquet as a DataFrame."""
    filename = file.filename.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file.file)
    elif filename.endswith(".parquet"):
        df = pd.read_parquet(file.file)
    else:
        raise ValueError("Unsupported data format")
    file.file.seek(0)
    return df


def extract_text_from_html(file: UploadFile) -> str:
    """Extract visible text from HTML."""
    soup = BeautifulSoup(file.file.read(), "html.parser")
    file.file.seek(0)
    return soup.get_text(separator=" ")


def extract_text_generic(file: UploadFile) -> str:
    """Fallback: treat file as plain text."""
    try:
        return file.file.read().decode("utf-8", errors="ignore")
    except Exception:
        return "[Unsupported file format or unreadable binary data]"


# ---------- API Endpoint ----------

@app.post("/api/")
async def query_api(
    question: UploadFile = File(None),
    image: UploadFile = File(None),
    data: UploadFile = File(None),
    attachments: list[UploadFile] = File(None)
):
    context_parts = []
    dataframes = []

    # Handle explicitly named params
    if question:
        context_parts.append(extract_text_generic(question))
    if image:
        context_parts.append(extract_text_from_image(image))
    if data:
        try:
            df = load_dataframe(data)
            dataframes.append(df)
            context_parts.append(f"Data file '{data.filename}' loaded with shape {df.shape}.")
        except Exception:
            context_parts.append(extract_text_generic(data))

    # Handle multiple generic attachments
    if attachments:
        for f in attachments:
            fname = f.filename.lower()
            if fname.endswith((".csv", ".xls", ".xlsx", ".parquet")):
                try:
                    df = load_dataframe(f)
                    dataframes.append(df)
                    context_parts.append(f"Data file '{f.filename}' loaded with shape {df.shape}.")
                except Exception:
                    context_parts.append(extract_text_generic(f))
            elif fname.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                context_parts.append(extract_text_from_image(f))
            elif fname.endswith((".html", ".htm")):
                context_parts.append(extract_text_from_html(f))
            else:
                context_parts.append(extract_text_generic(f))

    # Register dataframes into DuckDB for querying
    for idx, df in enumerate(dataframes):
        duckdb.register(f"df{idx}", df)

    # Combine all contexts
    context_text = "\n".join(context_parts) if context_parts else ""

    # Prepare OpenAI payload
    payload = {
        "model": "openai/gpt-4o-mini",
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a data assistant. Use the files and datasets as context. If dataframes are available in DuckDB, you can suggest queries to analyze them."},
            {"role": "user", "content": f"{context_text}"}
        ]
    }

    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", OPENAI_API_URL, headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }, json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line.strip().startswith("data: "):
                        data = line[len("data: "):]
                        if data == "[DONE]":
                            break
                        yield data + "\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import io, base64, os, json, asyncio, httpx
from bs4 import BeautifulSoup
from typing import List, Optional, AsyncGenerator

app = FastAPI()

# --- Utility: Encode matplotlib figure as base64 string ---
def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# --- Heuristic file processors ---
def analyze_csv(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        summary = {
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "head": df.head().to_dict(orient="records"),
        }
        if df.shape[1] >= 2:
            df.iloc[:,0:2].plot()
            summary["plot"] = fig_to_base64()
        return summary
    except Exception as e:
        return {"error": str(e)}

def analyze_parquet(file: UploadFile):
    try:
        df = pd.read_parquet(file.file)
        return {
            "columns": df.columns.tolist(),
            "shape": df.shape,
        }
    except Exception as e:
        return {"error": str(e)}

# --- Streaming LLM agent planning via HTTPX ---
async def stream_agent_answer(question: str, context: dict) -> AsyncGenerator[bytes, None]:
    api_key = "replace with your own API key"
    tools_desc = """
    You can:
    - Use pandas/duckdb for tabular data
    - Use matplotlib for plots (return as base64)
    - Use BeautifulSoup/requests for web scraping
    Always return JSON. If multiple answers: return a JSON array, else a JSON object.
    """

    prompt = f"""
    The user asked: {question}
    Available context keys: {list(context.keys())}

    {tools_desc}
    """

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "https://aipipe.org/openrouter/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a data analyst agent."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0,
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[len("data: "):]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data)
                            delta = parsed["choices"][0]["delta"].get("content")
                            if delta:
                                yield delta.encode("utf-8")
                        except Exception:
                            continue
    except Exception as e:
        yield json.dumps({"error": str(e)}).encode("utf-8")

# --- Main endpoint ---
@app.post("/api/")
async def process_query(
    questions: UploadFile = File(..., description="questions.txt"),
    files: Optional[List[UploadFile]] = File(None)
):
    q_text = (await questions.read()).decode("utf-8", errors="ignore")
    context = {}

    if files:
        for f in files:
            if f.filename.endswith(".csv"):
                context[f.filename] = analyze_csv(f)
            elif f.filename.endswith(".parquet"):
                context[f.filename] = analyze_parquet(f)
            elif f.filename.endswith(".json"):
                try:
                    context[f.filename] = json.load(f.file)
                except Exception as e:
                    context[f.filename] = {"error": str(e)}
            else:
                context[f.filename] = {"note": "Unsupported file type"}

    return StreamingResponse(stream_agent_answer(q_text, context), media_type="text/plain")

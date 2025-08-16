# 📘 AI Agent API (FastAPI + OpenAI + Heuristics)

This project is a **FastAPI-based API** that acts as a lightweight AI agent.  
It uses **httpx** to communicate with the OpenAI API (streaming responses token-by-token), with some simple heuristics and knowledge retrieval helpers powered by **DuckDB** and **BeautifulSoup**.  

You can query the API with natural language questions (like those from course sites or example prompts) and get AI-generated answers in real time.

---

## 🚀 Features
- FastAPI backend with async endpoints
- Streams OpenAI responses **token by token** via `httpx`
- Lightweight heuristics (basic keyword routing before LLM call)
- Context augmentation via **DuckDB** (structured data) and **BeautifulSoup** (HTML parsing)
- Easy deployment and public access with **ngrok**
- Configured with `.env` for secrets (API keys)

---



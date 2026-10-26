# AasthaGPT

AI resume chatbot. FastAPI backend (`api/index.py`) + single-page frontend (`index.html`), deployed together on Vercel.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then put your real GROQ_API_KEY in .env
uvicorn api.index:app --reload
```
Open http://localhost:8000

## Deploy to Vercel
1. Push this folder to a GitHub repo (`.env` is git-ignored).
2. vercel.com -> Add New -> Project -> import the repo -> Framework Preset: Other.
3. Settings -> Environment Variables -> add `GROQ_API_KEY`.
4. Deploy. To update the resume, replace `A_Resume.pdf` and push.

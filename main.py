import os
import base64
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

from database import create_document, get_documents
from schemas import Wasteanalysis

app = FastAPI(title="EcoWaste - AI Waste Management API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeResponse(BaseModel):
    category: str
    confidence: float
    instructions: List[str]
    notes: Optional[str] = None


@app.get("/")
def read_root():
    return {"message": "EcoWaste API is running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = getattr(db, 'name', "✅ Connected")
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


def call_gemini_with_image(b64_data: str, mime_type: str) -> str:
    """Call Gemini API with an image and return text response."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing GOOGLE_API_KEY environment variable")

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "You are an expert in waste identification and eco-friendly dismantling. "
                            "Classify the waste in the image (e.g., plastic bottle, e-waste, metal can, glass jar, paper, organic, textile, battery, etc.). "
                            "Return a concise JSON with keys: category (string), confidence (0-1), instructions (array of 3-7 short imperative steps), notes (optional safety/environmental tips). "
                            "Focus on safe, sustainable handling and local recycling best practices."
                        )
                    },
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": b64_data,
                        }
                    }
                ]
            }
        ]
    }

    r = requests.post(url, params={"key": api_key}, json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {r.text[:200]}")

    data = r.json()
    # The response format: candidates[0].content.parts[0].text
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise HTTPException(status_code=502, detail="Unexpected Gemini response format")
    return text


def parse_gemini_output_to_result(text: str) -> AnalyzeResponse:
    """Attempt to parse model text into structured result. Falls back gracefully."""
    import json, re

    # Try to find JSON block in the text
    json_str = None
    # Look for a fenced code block or first JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        json_str = match.group(0)

    if json_str:
        try:
            obj = json.loads(json_str)
            category = str(obj.get("category") or obj.get("type") or "unknown").strip()
            confidence = float(obj.get("confidence", 0.7))
            instructions = obj.get("instructions") or obj.get("steps") or []
            if isinstance(instructions, str):
                instructions = [instructions]
            if not isinstance(instructions, list):
                instructions = []
            notes = obj.get("notes")
            return AnalyzeResponse(category=category or "unknown", confidence=max(0.0, min(1.0, confidence)), instructions=instructions[:10], notes=notes)
        except Exception:
            pass

    # Fallback: build a basic structure from plain text
    lines = [l.strip("- • ") for l in text.splitlines() if l.strip()]
    steps = [l for l in lines if len(l) > 3][:5]
    return AnalyzeResponse(category="unknown", confidence=0.6, instructions=steps or ["Rinse and sort by material", "Check local recycling guidelines", "Dispose responsibly"], notes=None)


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    # Validate mime type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    # Read and base64 encode
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    b64 = base64.b64encode(content).decode("utf-8")

    # Call Gemini
    text = call_gemini_with_image(b64, file.content_type)
    structured = parse_gemini_output_to_result(text)

    # Persist to DB
    doc = Wasteanalysis(
        filename=file.filename or "image",
        mime_type=file.content_type,
        category=structured.category,
        confidence=structured.confidence,
        instructions=structured.instructions,
        notes=structured.notes,
    )
    try:
        create_document("wasteanalysis", doc)
    except Exception:
        # Non-fatal if DB is not configured
        pass

    return structured


@app.get("/api/analyses")
def list_analyses(limit: int = 20):
    try:
        docs = get_documents("wasteanalysis", limit=limit)
        # Map to simple public structure
        results = []
        for d in docs:
            results.append({
                "filename": d.get("filename"),
                "mime_type": d.get("mime_type"),
                "category": d.get("category"),
                "confidence": d.get("confidence"),
                "instructions": d.get("instructions", []),
                "notes": d.get("notes"),
                "created_at": d.get("created_at"),
            })
        return {"items": results}
    except Exception:
        # DB not available
        return {"items": []}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

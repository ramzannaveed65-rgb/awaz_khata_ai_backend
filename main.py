import os
import json
import datetime
import io
import asyncio
import re
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from pydantic import BaseModel
import PIL.Image

# Local Imports
import models
from database import engine, get_db, Base

load_dotenv()
Base.metadata.create_all(bind=engine)
app = FastAPI(title="AwazKhata AI Backend 2026")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# --- DYNAMIC MODEL SELECTION ---
# Based on your terminal logs, we will prioritize 3.5-flash
MODEL_NAME = "models/gemini-3.5-flash" 

class VoiceInput(BaseModel):
    transcript: str

def clean_json_response(text):
    # Remove markdown code blocks if present
    cleaned = re.sub(r'```json|```', '', text).strip()
    return cleaned

@app.get("/")
def home():
    return {"status": "Online", "message": "AwazKhata API is active"}

# -------------------------------------------------------------------------
# 2. AI BILL SCANNER
# -------------------------------------------------------------------------
@app.post("/stock/scan-bill")
async def scan_bill(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        request_object_content = await file.read()
        image = PIL.Image.open(io.BytesIO(request_object_content))

        prompt = """
        Analyze this bill. Return a JSON list of objects.
        Required keys: "name", "qty", "unit", "price".
        Format: [{"name": "Item", "qty": 1.0, "unit": "pcs", "price": 0.0}]
        """

        response_text = ""
        
        # RETRY LOOP WITH THE CORRECT MODEL NAME
        for attempt in range(3):
            try:
                print(f"--- AI Scan Attempt {attempt + 1} using {MODEL_NAME} ---")
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[prompt, image],
                    config={'response_mime_type': 'application/json'}
                )
                if response and response.text:
                    response_text = clean_json_response(response.text)
                    print(f"🎯 AI Success!")
                    break 
            except Exception as e:
                err_msg = str(e)
                if "503" in err_msg or "429" in err_msg:
                    wait_time = (attempt + 1) * 2
                    print(f"⚠️ Google Busy. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"❌ AI Error: {err_msg}")
                    raise e

        if not response_text:
            raise HTTPException(status_code=503, detail="AI response empty.")

        items_from_bill = json.loads(response_text)
        results_summary = []
        items_for_flutter = []

        for entry in items_from_bill:
            item_name = entry.get("name", "Unknown").strip()
            qty = float(entry.get("qty", 0.0))
            unit = entry.get("unit", "pcs")
            price = float(entry.get("price", 0.0))

            db_item = db.query(models.Item).filter(models.Item.name.ilike(item_name)).first()
            if not db_item:
                db_item = models.Item(name=item_name, quantity=0.0, unit=unit)
                if hasattr(db_item, 'sale_price'): db_item.sale_price = price
                db.add(db_item)
                db.flush() 

            db_item.quantity += qty
            db.add(models.StockTransaction(item_id=db_item.id, type="in", quantity=qty))
            
            results_summary.append(f"Added {qty} {unit} of {item_name}")
            items_for_flutter.append({"name": item_name, "qty": qty, "unit": unit, "price": price})

        db.commit()
        return {"status": "success", "added_items": results_summary, "items_data": items_for_flutter}

    except Exception as e:
        db.rollback()
        print(f"❌ SCAN FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/inventory")
def get_inventory(db: Session = Depends(get_db)):
    items = db.query(models.Item).all()
    return [{
        "id": i.id, "name": i.name, "quantity": float(i.quantity), 
        "unit": i.unit, "price": float(getattr(i, 'sale_price', 0.0))
    } for i in items]
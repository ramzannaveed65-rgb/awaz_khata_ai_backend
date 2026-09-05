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

# 1. INITIALIZATION
load_dotenv()
Base.metadata.create_all(bind=engine)
app = FastAPI(title="AwazKhata AI Backend 2026")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# The model confirmed by your terminal logs
MODEL_NAME = "models/gemini-3.5-flash" 

class VoiceInput(BaseModel):
    transcript: str

def clean_json_response(text):
    """Strips markdown and ensures clean JSON parsing."""
    cleaned = re.sub(r'```json|```', '', text).strip()
    return cleaned

@app.get("/")
def home():
    return {"status": "Online", "message": "AwazKhata API is active"}

# -------------------------------------------------------------------------
# 2. AI BILL SCANNER (Image Processing)
# -------------------------------------------------------------------------
@app.post("/stock/scan-bill")
async def scan_bill(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        request_object_content = await file.read()
        image = PIL.Image.open(io.BytesIO(request_object_content))

        prompt = """
        Analyze this bill. Return a JSON list of products.
        Required keys: "name", "qty", "unit", "price".
        Note: Translate names to English (e.g., Namak to Salt).
        Format: [{"name": "Item", "qty": 1.0, "unit": "pcs", "price": 0.0}]
        """

        response_text = ""
        # Retry loop for 503 errors
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[prompt, image],
                    config={'response_mime_type': 'application/json'}
                )
                if response and response.text:
                    response_text = clean_json_response(response.text)
                    break 
            except Exception as e:
                if "503" in str(e) or "429" in str(e):
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                raise e

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
        print(f"❌ SCAN ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------------
# 3. MAIN VOICE PROCESSOR (Now with Retry Logic)
# -------------------------------------------------------------------------
@app.post("/voice/process")
async def process_voice(data: VoiceInput, db: Session = Depends(get_db)):
    prompt = f"""
    You are AwazKhata AI. Convert this command into JSON: "{data.transcript}"
    
    Translation Mapping:
    - Namak -> Salt, Chini/Shakar -> Sugar, Pani -> Water, Dudh -> Milk, Atta -> Flour.
    
    Intents:
    - If user asks 'kitna hai', 'bachi hai', 'stock dikhao' -> ACTION: "QUERY"
    - If user says 'add karo', 'le aya', 'khareeda' -> ACTION: "STOCK_IN"
    - If user says 'becha', 'sell', 'nikal do' -> ACTION: "STOCK_OUT"
    - If user asks 'aaj ki sale' -> ACTION: "SUMMARY"

    Return ONLY this JSON:
    {{
      "action": "STOCK_IN / STOCK_OUT / QUERY / SUMMARY",
      "item": "Standard English Name",
      "qty": 0.0,
      "unit": "kg / pcs / pack",
      "voice_response": "Natural Urdu response"
    }}
    """
    
    response_text = ""
    # --- ADDED RETRY LOGIC FOR VOICE ---
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, 
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            if response and response.text:
                response_text = clean_json_response(response.text)
                break
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                print(f"⚠️ Google Busy. Retrying voice in {attempt+2}s...")
                await asyncio.sleep(attempt + 2)
                continue
            raise e

    if not response_text:
        raise HTTPException(status_code=503, detail="AI is temporarily unavailable")

    try:
        intent = json.loads(response_text)
        action = intent.get("action")
        item_name = intent.get("item", "").strip()
        qty = float(intent.get("qty", 0.0))

        db_item = db.query(models.Item).filter(models.Item.name.ilike(item_name)).first()

        if action == "QUERY":
            if db_item:
                intent["voice_response"] = f"{db_item.name} ka stock {db_item.quantity} {db_item.unit} bacha hai."
            else:
                intent["voice_response"] = f"Maaf kijie, {item_name} record mein nahi mila."

        elif action in ["STOCK_IN", "STOCK_OUT"]:
            if not db_item and action == "STOCK_IN":
                db_item = models.Item(name=item_name, quantity=0.0, unit=intent.get("unit", "pcs"))
                db.add(db_item)
                db.flush()

            if db_item:
                if action == "STOCK_IN": db_item.quantity += qty
                else: db_item.quantity -= qty
                
                db.add(models.StockTransaction(item_id=db_item.id, type=action.lower().replace("stock_", ""), quantity=qty))
                db.commit()
            else:
                intent["voice_response"] = f"Pehle {item_name} ko database mein add karein."

        elif action == "SUMMARY":
            today = datetime.date.today()
            sales = db.query(models.StockTransaction).filter(
                models.StockTransaction.type == 'out',
                models.StockTransaction.timestamp >= today
            ).all()
            total = sum(s.quantity for s in sales)
            intent["voice_response"] = f"Aaj ki total sale {total} units hai."

        return intent

    except Exception as e:
        db.rollback()
        print(f"❌ VOICE LOGIC ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process voice intent")

# -------------------------------------------------------------------------
# 4. UTILITY ENDPOINTS
# -------------------------------------------------------------------------
@app.get("/inventory")
def get_inventory(db: Session = Depends(get_db)):
    try:
        items = db.query(models.Item).all()
        return [{
            "id": i.id, 
            "name": i.name, 
            "quantity": float(i.quantity), 
            "unit": i.unit, 
            "price": float(getattr(i, 'sale_price', 0.0))
        } for i in items]
    except Exception as e:
        print(f"❌ INVENTORY ERROR: {e}")
        return []
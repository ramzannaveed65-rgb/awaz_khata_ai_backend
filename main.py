import os
import json
import datetime
import io
import asyncio
import re
import traceback
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

MODEL_NAME = "models/gemini-3.5-flash"


class VoiceInput(BaseModel):
    transcript: str
    confirm: bool = False
    actions: list | None = None


def clean_json_response(text):
    """Strips markdown and ensures clean JSON parsing."""
    cleaned = re.sub(r'```json|```', '', text).strip()
    return cleaned


def normalize_actions(parsed):
    """Gemini may return one object or a list. Always return a list."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if isinstance(parsed.get("actions"), list):
            return parsed["actions"]
        return [parsed]
    return []


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

        items_from_bill = normalize_actions(json.loads(response_text))
        results_summary = []
        items_for_flutter = []

        for entry in items_from_bill:
            item_name = entry.get("name", "Unknown").strip()
            qty = float(entry.get("qty", 0.0))
            unit = entry.get("unit", "pcs")
            price = float(entry.get("price", 0.0))

            db_item = db.query(models.Item).filter(
                models.Item.name.ilike(item_name)).first()
            if not db_item:
                db_item = models.Item(name=item_name, quantity=0.0, unit=unit)
                if hasattr(db_item, 'sale_price'):
                    db_item.sale_price = price
                db.add(db_item)
                db.flush()

            db_item.quantity += qty
            db.add(models.StockTransaction(
                item_id=db_item.id, type="in", quantity=qty))

            results_summary.append(f"Added {qty} {unit} of {item_name}")
            items_for_flutter.append(
                {"name": item_name, "qty": qty, "unit": unit, "price": price})

        db.commit()
        return {
            "status": "success",
            "added_items": results_summary,
            "items_data": items_for_flutter,
        }
    except Exception as e:
        db.rollback()
        print(f"❌ SCAN ERROR: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------------
# 3. MAIN VOICE PROCESSOR — parse (no writes) then confirm (writes)
# -------------------------------------------------------------------------
@app.post("/voice/process")
async def process_voice(data: VoiceInput, db: Session = Depends(get_db)):
    # ---------- PHASE 2: user confirmed → write to DB ----------
    if data.confirm:
        return execute_actions(data.actions or [], db)

    # ---------- PHASE 1: parse only, never writes ----------
    prompt = f"""
    You are AwazKhata AI. Convert this command into JSON: "{data.transcript}"

    IMPORTANT: The user may give MORE THAN ONE command in a single sentence,
    joined by "aur", "or", "and", or a pause.
    Example: "5 kilo cheeni add kro aur 5 kilo namak add kro" = TWO commands.
    Return one JSON object per command.

    Translation Mapping:
    - Namak -> Salt, Chini/Shakar -> Sugar, Pani -> Water, Dudh -> Milk, Atta -> Flour.

    Intents:
    - 'kitna hai', 'bachi hai', 'stock dikhao' -> "QUERY"
    - 'add karo', 'le aya', 'khareeda' -> "STOCK_IN"
    - 'becha', 'bech di', 'sell', 'nikal do' -> "STOCK_OUT"
    - 'aaj ki sale' -> "SUMMARY"

    Return ONLY a JSON ARRAY, even when there is just one command:
    [
      {{
        "action": "STOCK_IN / STOCK_OUT / QUERY / SUMMARY",
        "item": "Standard English Name",
        "qty": 0.0,
        "unit": "kg / pcs / pack",
        "voice_response": "Natural Urdu response"
      }}
    ]
    """

    response_text = ""
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
        actions = normalize_actions(json.loads(response_text))
        if not actions:
            raise HTTPException(status_code=422, detail="Could not understand command")

        for a in actions:
            action = a.get("action")
            item_name = (a.get("item") or "").strip()
            qty = float(a.get("qty") or 0.0)

            if action == "QUERY":
                db_item = db.query(models.Item).filter(
                    models.Item.name.ilike(item_name)).first()
                a["voice_response"] = (
                    f"{db_item.name} ka stock {db_item.quantity} {db_item.unit} bacha hai."
                    if db_item else
                    f"Maaf kijie, {item_name} record mein nahi mila."
                )

            elif action == "SUMMARY":
                today = datetime.date.today()
                sales = db.query(models.StockTransaction).filter(
                    models.StockTransaction.type == 'out',
                    models.StockTransaction.timestamp >= today
                ).all()
                total = sum(s.quantity for s in sales)
                a["voice_response"] = f"Aaj ki total sale {total} units hai."

            elif action == "STOCK_OUT":
                db_item = db.query(models.Item).filter(
                    models.Item.name.ilike(item_name)).first()
                if not db_item:
                    a["warning"] = f"{item_name} stock mein nahi hai."
                elif db_item.quantity < qty:
                    a["warning"] = (
                        f"Sirf {db_item.quantity} {db_item.unit} {item_name} bacha hai."
                    )

        needs_confirm = any(
            a.get("action") in ("STOCK_IN", "STOCK_OUT") for a in actions
        )
        return {"actions": actions, "needs_confirm": needs_confirm}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ VOICE LOGIC ERROR: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to process voice intent")


def execute_actions(actions, db):
    """Commits confirmed STOCK_IN / STOCK_OUT actions. All-or-nothing."""
    results = []
    try:
        for a in actions:
            action = a.get("action")
            if action not in ("STOCK_IN", "STOCK_OUT"):
                continue

            item_name = (a.get("item") or "").strip()
            qty = float(a.get("qty") or 0.0)
            if qty <= 0:
                raise ValueError(f"Invalid quantity for {item_name}")

            db_item = db.query(models.Item).filter(
                models.Item.name.ilike(item_name)).first()

            if not db_item:
                if action == "STOCK_OUT":
                    raise ValueError(f"{item_name} stock mein nahi hai")
                db_item = models.Item(
                    name=item_name, quantity=0.0, unit=a.get("unit", "pcs"))
                db.add(db_item)
                db.flush()

            if action == "STOCK_IN":
                db_item.quantity += qty
            else:
                if db_item.quantity < qty:
                    raise ValueError(
                        f"Sirf {db_item.quantity} {db_item.unit} {item_name} bacha hai")
                db_item.quantity -= qty

            db.add(models.StockTransaction(
                item_id=db_item.id,
                type=action.lower().replace("stock_", ""),
                quantity=qty,
            ))
            results.append(f"{item_name}: {db_item.quantity} {db_item.unit}")

        db.commit()
        return {"status": "success", "results": results}

    except Exception as e:
        db.rollback()
        print(f"❌ EXECUTE ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


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
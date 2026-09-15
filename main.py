import os
import json
import datetime
import io
import asyncio
import re
import logging
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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("awazkhata")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

MODEL_NAME = "models/gemini-3.5-flash-lite"

# Largest edge (in pixels) sent to Gemini. Phone photos are typically
# 3000px+, which costs upload time and tokens for no accuracy gain on bills.
MAX_IMAGE_EDGE = 1600

# Retry tuning for "Google Busy" (503) and rate-limit (429) responses.
AI_ATTEMPTS = 4
AI_BACKOFF = [2, 4, 8]  # seconds between attempts

BUSY_MARKERS = ("503", "429", "overload", "unavailable", "quota",
                "rate limit", "resource_exhausted")


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


def _is_busy_error(err):
    msg = str(err).lower()
    return any(marker in msg for marker in BUSY_MARKERS)


async def ai_generate(contents, label="ai"):
    """
    Single place where Gemini is called.

    Two things this does that the old inline loops did not:

    1. Runs the SDK call in a worker thread. client.models.generate_content
       is synchronous — calling it directly inside `async def` blocks the
       whole event loop, so every other request (inventory polls, a second
       voice command) stalls until Gemini answers.

    2. Never returns an empty string. If every attempt fails or comes back
       blank it raises 503, so callers can never hand "" to json.loads().
    """
    last_error = None

    for attempt in range(AI_ATTEMPTS):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL_NAME,
                contents=contents,
                config={'response_mime_type': 'application/json'},
            )
            if response and response.text and response.text.strip():
                return clean_json_response(response.text)

            last_error = "empty response"
            log.warning("%s: empty model response (attempt %s/%s)",
                        label, attempt + 1, AI_ATTEMPTS)

        except Exception as e:
            if not _is_busy_error(e):
                # A real bug (bad API key, malformed request). Surface it
                # immediately instead of burning retries on it.
                raise
            last_error = str(e)
            log.warning("%s: Google busy, attempt %s/%s",
                        label, attempt + 1, AI_ATTEMPTS)

        if attempt < AI_ATTEMPTS - 1:
            await asyncio.sleep(AI_BACKOFF[attempt])

    log.error("%s: all %s attempts failed. Last: %s",
              label, AI_ATTEMPTS, last_error)
    raise HTTPException(
        status_code=503,
        detail="AI service is busy right now. Please try again in a moment.",
    )


def parse_ai_json(text, label="ai"):
    """Parse model output, logging the raw text when it is not valid JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.error("%s: model returned non-JSON: %r", label, text[:500])
        raise HTTPException(
            status_code=502,
            detail="AI returned an unreadable response. Please try again.",
        )


def prepare_image(raw_bytes):
    """Decode, flatten and downscale the uploaded bill before sending it."""
    image = PIL.Image.open(io.BytesIO(raw_bytes))

    # Strip alpha / palette so JPEG-style encoding downstream is safe.
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    if max(image.size) > MAX_IMAGE_EDGE:
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE),
                        PIL.Image.Resampling.LANCZOS)

    return image


@app.get("/")
def home():
    return {"status": "Online", "message": "AwazKhata API is active"}


@app.get("/health")
def health():
    """Cheap endpoint the Flutter app can ping on launch to warm the server."""
    return {"ok": True}


# -------------------------------------------------------------------------
# 2. AI BILL SCANNER (Image Processing)
# -------------------------------------------------------------------------
@app.post("/stock/scan-bill")
async def scan_bill(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        request_object_content = await file.read()
        image = prepare_image(request_object_content)

        prompt = """
        Analyze this bill. Return a JSON list of products.
        Required keys: "name", "qty", "unit", "price".
        Note: Translate names to English (e.g., Namak to Salt).
        Format: [{"name": "Item", "qty": 1.0, "unit": "pcs", "price": 0.0}]
        """

        response_text = await ai_generate([prompt, image], label="scan-bill")
        items_from_bill = normalize_actions(
            parse_ai_json(response_text, label="scan-bill"))

        if not items_from_bill:
            raise HTTPException(
                status_code=422,
                detail="No items could be read from this bill. "
                       "Try a clearer photo.",
            )

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

    except HTTPException:
        # Let 503 / 502 / 422 through with their real status code instead of
        # flattening everything into a 500.
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        log.error("SCAN ERROR: %s", e)
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

    response_text = await ai_generate(prompt, label="voice")

    try:
        actions = normalize_actions(parse_ai_json(response_text, label="voice"))
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
                today = datetime.datetime.combine(
                    datetime.date.today(), datetime.time.min)
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
        log.error("VOICE LOGIC ERROR: %s", e)
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
        log.error("EXECUTE ERROR: %s", e)
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
        log.error("INVENTORY ERROR: %s", e)
        return []
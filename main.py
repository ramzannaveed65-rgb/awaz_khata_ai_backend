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
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Query
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from pydantic import BaseModel
import PIL.Image

# Local Imports
import models
from database import engine, get_db, Base

# 1. INITIALIZATION
load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("awazkhata")

Base.metadata.create_all(bind=engine)

# Pakistan Standard Time. Timestamps are stored in UTC, but "today's sale"
# must mean a Pakistani calendar day, not a UTC one — otherwise every sale
# made after 7pm local lands in the next day's report.
PKT = datetime.timezone(datetime.timedelta(hours=5))


def _add_column_if_missing(inspector, table, column, ddl_type):
    """create_all() makes missing TABLES but never adds a column to a table
    that already exists. This fills that gap. Safe to run on every boot."""
    if not inspector.has_table(table):
        return
    existing = {c["name"] for c in inspector.get_columns(table)}
    if column in existing:
        return
    with engine.begin() as conn:
        conn.execute(text(
            f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
    log.info("schema: added %s.%s", table, column)


def ensure_schema():
    inspector = inspect(engine)
    _add_column_if_missing(inspector, "items", "sale_price", "FLOAT DEFAULT 0.0")
    _add_column_if_missing(inspector, "items", "cost_price", "FLOAT DEFAULT 0.0")
    _add_column_if_missing(inspector, "transactions", "unit_price",
                           "FLOAT DEFAULT 0.0")
    _add_column_if_missing(inspector, "transactions", "total_amount",
                           "FLOAT DEFAULT 0.0")


ensure_schema()

app = FastAPI(title="AwazKhata AI Backend 2026")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

MODEL_NAME = "models/gemini-3.5-flash-lite"

MAX_IMAGE_EDGE = 1600

AI_ATTEMPTS = 4
AI_BACKOFF = [2, 4, 8]

BUSY_MARKERS = ("503", "429", "overload", "unavailable", "quota",
                "rate limit", "resource_exhausted")

# A single line on a kiryana bill should never legitimately exceed this.
# Anything above it means a unit/price mix-up, not a real purchase.
MAX_SANE_LINE_AMOUNT = 500_000


class VoiceInput(BaseModel):
    transcript: str
    confirm: bool = False
    actions: list | None = None


# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------
def pkt_day_bounds(day=None):
    """Start and end of a Pakistani calendar day, as naive UTC datetimes
    matching how timestamps are stored."""
    if day is None:
        day = datetime.datetime.now(PKT).date()
    start_local = datetime.datetime.combine(day, datetime.time.min, tzinfo=PKT)
    end_local = start_local + datetime.timedelta(days=1)
    to_utc = lambda d: d.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return to_utc(start_local), to_utc(end_local), day


def clean_json_response(text_in):
    return re.sub(r'```json|```', '', text_in).strip()


def normalize_actions(parsed):
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if isinstance(parsed.get("actions"), list):
            return parsed["actions"]
        return [parsed]
    return []


def to_float(value, default=0.0):
    """Gemini sometimes returns '1,450' or 'Rs 185' instead of a number."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r'[^0-9.\-]', '', str(value))
    try:
        return float(cleaned) if cleaned else default
    except ValueError:
        return default


def _is_busy_error(err):
    msg = str(err).lower()
    return any(marker in msg for marker in BUSY_MARKERS)


async def ai_generate(contents, label="ai"):
    """Single place where Gemini is called. Runs the synchronous SDK call in
    a worker thread so it does not block the event loop, retries on busy
    errors, and never returns an empty string."""
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


def parse_ai_json(text_in, label="ai"):
    try:
        return json.loads(text_in)
    except json.JSONDecodeError:
        log.error("%s: model returned non-JSON: %r", label, text_in[:500])
        raise HTTPException(
            status_code=502,
            detail="AI returned an unreadable response. Please try again.",
        )


def prepare_image(raw_bytes):
    image = PIL.Image.open(io.BytesIO(raw_bytes))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    if max(image.size) > MAX_IMAGE_EDGE:
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE),
                        PIL.Image.Resampling.LANCZOS)
    return image


def normalize_pack_pricing(item_name, qty, unit, price, line_amount):
    """
    Bills price packaged goods by the PACK, not by the gram or millilitre.
    "Tea Leaves | 950 gm | 1,450 | 1,450" means Rs 1450 for one 950gm pack —
    NOT Rs 1450 per gram. Multiplying gives Rs 1,377,500, which is how a
    single tea row came to be 99% of a day's purchases.

    The tell is that RATE and AMOUNT are the same number: that only happens
    when the quantity being priced is one of something. When we see it, we
    fold the pack size into the name and store a single pack.

    Returns (qty, unit, price, name).
    """
    if qty > 1 and line_amount > 0 and abs(line_amount - price) < 0.01:
        packed_name = item_name
        size = f"{qty:g}{unit}".replace(" ", "")
        if size.lower() not in item_name.lower().replace(" ", ""):
            packed_name = f"{item_name} {qty:g}{unit}"
        log.info("scan-bill: %r priced per pack, storing 1 pack at %s",
                 item_name, price)
        return 1.0, "pack", price, packed_name

    # Second guard: a per-unit rate that produces an absurd line total.
    if qty > 1 and price > 0 and qty * price > MAX_SANE_LINE_AMOUNT:
        log.warning("scan-bill: %r line total %s is implausible, treating "
                    "the price as a pack price", item_name, qty * price)
        return 1.0, "pack", price, f"{item_name} {qty:g}{unit}"

    return qty, unit, price, item_name


@app.get("/")
def home():
    return {"status": "Online", "message": "AwazKhata API is active"}


@app.get("/health")
def health():
    return {"ok": True}


# -------------------------------------------------------------------------
# 2. AI BILL SCANNER — purchases (stock IN)
# -------------------------------------------------------------------------
@app.post("/stock/scan-bill")
async def scan_bill(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        request_object_content = await file.read()
        image = prepare_image(request_object_content)

        prompt = """
        Analyze this bill. Return a JSON list of products.
        Required keys: "name", "qty", "unit", "price", "amount".

        "price" is the PER-UNIT RATE column.
        "amount" is the LINE TOTAL column.
        If the bill shows only one price column, put the same number in both.

        IMPORTANT — packaged goods:
        If a row's RATE and AMOUNT are the same number, that price is for
        ONE WHOLE PACK, not per gram or per millilitre. In that case return
        qty=1, unit="pack", and put the pack size in the name.
        Example: "Tea Leaves | 950 gm | 1,450 | 1,450" must become
        {"name": "Tea Leaves 950gm", "qty": 1, "unit": "pack",
         "price": 1450, "amount": 1450}
        NOT qty=950 with price=1450, which would mean Rs 1,377,500.

        Return numbers as plain numbers: 1450, not "1,450" or "Rs 1450".
        Ignore any TOTAL, SUBTOTAL, CASH, CHANGE, TAX or DISCOUNT rows —
        those are not products.
        Translate names to English (e.g., Namak to Salt, Chini to Sugar).

        Format: [{"name": "Item", "qty": 1.0, "unit": "pcs",
                  "price": 0.0, "amount": 0.0}]
        """

        response_text = await ai_generate([prompt, image], label="scan-bill")
        items_from_bill = normalize_actions(
            parse_ai_json(response_text, label="scan-bill"))

        if not items_from_bill:
            raise HTTPException(
                status_code=422,
                detail="No items could be read from this bill. "
                       "Try a clearer photo.")

        results_summary = []
        items_for_flutter = []

        for entry in items_from_bill:
            item_name = str(entry.get("name", "Unknown")).strip()
            qty = to_float(entry.get("qty"))
            unit = entry.get("unit") or "pcs"
            price = to_float(entry.get("price"))
            line_amount = to_float(entry.get("amount"))

            if not item_name or qty <= 0:
                log.warning("scan-bill: skipping bad row %r", entry)
                continue

            # Catch pack-priced rows before they multiply out to nonsense.
            qty, unit, price, item_name = normalize_pack_pricing(
                item_name, qty, unit, price, line_amount)

            db_item = db.query(models.Item).filter(
                models.Item.name.ilike(item_name)).first()

            if not db_item:
                db_item = models.Item(name=item_name, quantity=0.0, unit=unit)
                db.add(db_item)
                db.flush()

            # A purchase bill shows what the shopkeeper PAID — that is
            # cost_price. Selling price is set separately.
            if price > 0:
                db_item.cost_price = price
                # First time we see this item there is no sale price yet,
                # so seed it from cost. The shopkeeper can change it later.
                if not db_item.sale_price:
                    db_item.sale_price = price

            db_item.quantity += qty
            db.add(models.StockTransaction(
                item_id=db_item.id,
                type="in",
                quantity=qty,
                unit_price=price,
                total_amount=round(qty * price, 2),
            ))

            results_summary.append(f"Added {qty:g} {unit} of {item_name}")
            items_for_flutter.append({
                "name": item_name,
                "qty": qty,
                "unit": unit,
                "price": float(db_item.cost_price or 0.0),
            })

        if not items_for_flutter:
            raise HTTPException(
                status_code=422,
                detail="No usable items were read from this bill.")

        db.commit()
        return {
            "status": "success",
            "added_items": results_summary,
            "items_data": items_for_flutter,
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        log.error("SCAN ERROR: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------------
# 3. MAIN VOICE PROCESSOR
# -------------------------------------------------------------------------
@app.post("/voice/process")
async def process_voice(data: VoiceInput, db: Session = Depends(get_db)):
    if data.confirm:
        return execute_actions(data.actions or [], db)

    prompt = f"""
    You are AwazKhata AI. Convert this command into JSON: "{data.transcript}"

    IMPORTANT: The user may give MORE THAN ONE command in a single sentence,
    joined by "aur", "or", "and", or a pause.
    Example: "5 kilo cheeni add kro aur 5 kilo namak add kro" = TWO commands.
    Return one JSON object per command.

    If the user states a price ("200 rupay kilo", "150 ka becha"), put the
    PER-UNIT price in "price". If no price is mentioned, use 0.

    Translation Mapping:
    - Namak -> Salt, Chini/Shakar -> Sugar, Pani -> Water, Dudh -> Milk, Atta -> Flour.

    Intents:
    - 'kitna hai', 'bachi hai', 'stock dikhao' -> "QUERY"
    - 'add karo', 'le aya', 'khareeda' -> "STOCK_IN"
    - 'becha', 'bech di', 'sell', 'nikal do' -> "STOCK_OUT"
    - 'aaj ki sale', 'aaj ka hisab' -> "SUMMARY"

    Return ONLY a JSON ARRAY, even when there is just one command:
    [
      {{
        "action": "STOCK_IN / STOCK_OUT / QUERY / SUMMARY",
        "item": "Standard English Name",
        "qty": 0.0,
        "unit": "kg / pcs / pack",
        "price": 0.0,
        "voice_response": "Natural Urdu response"
      }}
    ]
    """

    response_text = await ai_generate(prompt, label="voice")

    try:
        actions = normalize_actions(parse_ai_json(response_text, label="voice"))
        if not actions:
            raise HTTPException(status_code=422,
                                detail="Could not understand command")

        for a in actions:
            action = a.get("action")
            item_name = (a.get("item") or "").strip()
            qty = to_float(a.get("qty"))

            if action == "QUERY":
                db_item = db.query(models.Item).filter(
                    models.Item.name.ilike(item_name)).first()
                a["voice_response"] = (
                    f"{db_item.name} ka stock {db_item.quantity:g} {db_item.unit} bacha hai."
                    if db_item else
                    f"Maaf kijie, {item_name} record mein nahi mila."
                )

            elif action == "SUMMARY":
                start, end, day = pkt_day_bounds()
                sales = db.query(models.StockTransaction).filter(
                    models.StockTransaction.type == 'out',
                    models.StockTransaction.timestamp >= start,
                    models.StockTransaction.timestamp < end,
                ).all()
                revenue = sum(s.total_amount or 0.0 for s in sales)
                a["total_sale"] = round(revenue, 2)
                a["bill_count"] = len(sales)
                a["voice_response"] = (
                    f"Aaj ki total sale {revenue:,.0f} rupay hai, "
                    f"{len(sales)} items bikay."
                    if sales else "Aaj abhi tak koi sale nahi hui."
                )

            elif action == "STOCK_OUT":
                db_item = db.query(models.Item).filter(
                    models.Item.name.ilike(item_name)).first()
                if not db_item:
                    a["warning"] = f"{item_name} stock mein nahi hai."
                elif db_item.quantity < qty:
                    a["warning"] = (
                        f"Sirf {db_item.quantity:g} {db_item.unit} {item_name} bacha hai.")
                else:
                    # Show the shopkeeper what this sale will come to
                    # BEFORE they confirm it.
                    rate = to_float(a.get("price")) or float(db_item.sale_price or 0.0)
                    a["unit_price"] = rate
                    a["line_total"] = round(qty * rate, 2)

        needs_confirm = any(
            a.get("action") in ("STOCK_IN", "STOCK_OUT") for a in actions)
        return {"actions": actions, "needs_confirm": needs_confirm}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error("VOICE LOGIC ERROR: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500,
                            detail="Failed to process voice intent")


def execute_actions(actions, db):
    """Commits confirmed STOCK_IN / STOCK_OUT actions. All-or-nothing."""
    results = []
    sale_total = 0.0
    try:
        for a in actions:
            action = a.get("action")
            if action not in ("STOCK_IN", "STOCK_OUT"):
                continue

            item_name = (a.get("item") or "").strip()
            qty = to_float(a.get("qty"))
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

            spoken_price = to_float(a.get("price"))

            if action == "STOCK_IN":
                rate = spoken_price or float(db_item.cost_price or 0.0)
                if spoken_price > 0:
                    db_item.cost_price = spoken_price
                db_item.quantity += qty
            else:
                if db_item.quantity < qty:
                    raise ValueError(
                        f"Sirf {db_item.quantity:g} {db_item.unit} {item_name} bacha hai")
                # Sale price: what was spoken, else the item's standing price.
                rate = spoken_price or float(db_item.sale_price or 0.0)
                if spoken_price > 0:
                    db_item.sale_price = spoken_price
                db_item.quantity -= qty
                sale_total += qty * rate

            line_total = round(qty * rate, 2)

            db.add(models.StockTransaction(
                item_id=db_item.id,
                type=action.lower().replace("stock_", ""),
                quantity=qty,
                unit_price=rate,
                total_amount=line_total,
            ))

            # Say what HAPPENED, then what is left. The old wording put the
            # remaining stock next to the sale amount, which read as though
            # 1 kg of sugar had sold for Rs 370.
            if action == "STOCK_OUT":
                results.append(
                    f"Sold {qty:g} {db_item.unit} {item_name} — "
                    f"Rs {line_total:,.0f} ({db_item.quantity:g} {db_item.unit} left)")
            else:
                results.append(
                    f"Added {qty:g} {db_item.unit} {item_name} "
                    f"({db_item.quantity:g} {db_item.unit} in stock)")

        db.commit()
        return {
            "status": "success",
            "results": results,
            "sale_total": round(sale_total, 2),
        }

    except Exception as e:
        db.rollback()
        log.error("EXECUTE ERROR: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------------------------------------------------
# 4. REPORTS
# -------------------------------------------------------------------------
@app.get("/reports/daily")
def daily_report(
    date: str | None = Query(None, description="YYYY-MM-DD, defaults to today"),
    db: Session = Depends(get_db),
):
    """Day-end sales record: what sold, at what price, and the day's total."""
    try:
        day = None
        if date:
            try:
                day = datetime.datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400,
                                    detail="date must be YYYY-MM-DD")

        start, end, day = pkt_day_bounds(day)

        rows = (
            db.query(models.StockTransaction, models.Item)
            .join(models.Item, models.Item.id == models.StockTransaction.item_id)
            .filter(models.StockTransaction.timestamp >= start,
                    models.StockTransaction.timestamp < end)
            .order_by(models.StockTransaction.timestamp)
            .all()
        )

        sales, purchases = [], []
        revenue = cost = 0.0

        for txn, item in rows:
            amount = float(txn.total_amount or 0.0)
            record = {
                "time": (txn.timestamp.replace(tzinfo=datetime.timezone.utc)
                         .astimezone(PKT).strftime("%H:%M")),
                "item": item.name,
                "qty": float(txn.quantity or 0.0),
                "unit": item.unit,
                "unit_price": float(txn.unit_price or 0.0),
                "amount": amount,
            }
            if txn.type == "out":
                sales.append(record)
                revenue += amount
            else:
                purchases.append(record)
                cost += amount

        # Per-item roll-up so the shopkeeper sees what moved, not 40 lines.
        by_item = {}
        for s in sales:
            b = by_item.setdefault(
                s["item"], {"item": s["item"], "unit": s["unit"],
                            "qty": 0.0, "amount": 0.0})
            b["qty"] += s["qty"]
            b["amount"] += s["amount"]

        return {
            "date": day.isoformat(),
            "total_sale": round(revenue, 2),
            "total_purchase": round(cost, 2),
            "sale_count": len(sales),
            "items_sold": sorted(by_item.values(),
                                 key=lambda x: x["amount"], reverse=True),
            "sales": sales,
            "purchases": purchases,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error("DAILY REPORT ERROR: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to build report")


@app.get("/reports/range")
def range_report(
    days: int = Query(7, ge=1, le=90, description="How many days back"),
    db: Session = Depends(get_db),
):
    """Sale totals per day, newest first. For a weekly/monthly chart."""
    try:
        today = datetime.datetime.now(PKT).date()
        out = []
        for i in range(days):
            day = today - datetime.timedelta(days=i)
            start, end, _ = pkt_day_bounds(day)
            rows = db.query(models.StockTransaction).filter(
                models.StockTransaction.type == 'out',
                models.StockTransaction.timestamp >= start,
                models.StockTransaction.timestamp < end,
            ).all()
            out.append({
                "date": day.isoformat(),
                "total_sale": round(sum(r.total_amount or 0.0 for r in rows), 2),
                "sale_count": len(rows),
            })
        return {"days": days, "report": out}
    except Exception as e:
        log.error("RANGE REPORT ERROR: %s", e)
        raise HTTPException(status_code=500, detail="Failed to build report")


# -------------------------------------------------------------------------
# 5. UTILITY ENDPOINTS
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
            "min_stock": float(i.min_stock or 0.0),
            "price": float(i.sale_price or 0.0),
            "cost_price": float(i.cost_price or 0.0),
        } for i in items]
    except Exception as e:
        log.error("INVENTORY ERROR: %s", e)
        return []


class PriceUpdate(BaseModel):
    sale_price: float


@app.put("/items/{item_id}/price")
def set_sale_price(item_id: int, body: PriceUpdate,
                   db: Session = Depends(get_db)):
    """Lets the shopkeeper set a selling price that differs from what the
    purchase bill said. Without this, every sale records at cost."""
    item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    if body.sale_price < 0:
        raise HTTPException(status_code=400, detail="Price cannot be negative")
    item.sale_price = body.sale_price
    db.commit()
    return {"status": "success", "id": item.id, "name": item.name,
            "sale_price": float(item.sale_price)}


class ItemFix(BaseModel):
    quantity: float | None = None
    unit: str | None = None
    cost_price: float | None = None
    sale_price: float | None = None


@app.put("/items/{item_id}")
def fix_item(item_id: int, body: ItemFix, db: Session = Depends(get_db)):
    """Correct an item the scanner got wrong — wrong unit, wrong quantity,
    wrong price. Needed because a bad scan otherwise stays bad forever."""
    item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    if body.quantity is not None:
        if body.quantity < 0:
            raise HTTPException(status_code=400,
                                detail="Quantity cannot be negative")
        item.quantity = body.quantity
    if body.unit:
        item.unit = body.unit
    if body.cost_price is not None:
        item.cost_price = body.cost_price
    if body.sale_price is not None:
        item.sale_price = body.sale_price

    db.commit()
    return {"status": "success", "id": item.id, "name": item.name,
            "quantity": float(item.quantity), "unit": item.unit,
            "cost_price": float(item.cost_price or 0.0),
            "sale_price": float(item.sale_price or 0.0)}
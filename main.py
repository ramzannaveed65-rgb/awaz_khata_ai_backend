import os
import json
import datetime
import io
import asyncio
import re
import difflib
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
MAX_SANE_LINE_AMOUNT = 500_000

# How alike two names must be before they are treated as the same thing.
# Scans use the stricter value because wrongly merging two products
# corrupts stock counts permanently.
FUZZY_CUTOFF_VOICE = 0.82
FUZZY_CUTOFF_SCAN = 0.90

# Customer names are shorter and more varied than product names, so a
# slightly looser threshold works better — "Ali" vs "Ali Bhai".
FUZZY_CUTOFF_CUSTOMER = 0.80


class VoiceInput(BaseModel):
    transcript: str
    confirm: bool = False
    actions: list | None = None


# -------------------------------------------------------------------------
# NAME MATCHING
# -------------------------------------------------------------------------
def _norm(s):
    """Lowercase, strip punctuation, collapse spaces."""
    s = re.sub(r'[^a-z0-9 ]', ' ', (s or '').lower())
    return re.sub(r'\s+', ' ', s).strip()


def _squash(s):
    """Normalised form with doubled letters collapsed and spaces removed.

    This is what catches Chilli vs Chili, Yoghurt vs Yogurt, Dal vs Daal —
    the spellings Gemini alternates between for the same thing.
    """
    return re.sub(r'(.)\1+', r'\1', _norm(s).replace(' ', ''))


def find_by_name(db, model, name, allow_partial=False,
                 cutoff=FUZZY_CUTOFF_VOICE):
    """
    Look up a row by name, tolerating the spelling drift that comes from
    passing names through a language model and a speech recogniser.

    Tried in order, stopping at the first hit:
      1. exact match (case-insensitive)
      2. squashed match — handles doubled-letter spellings
      3. close match by edit distance
      4. partial match, and only when the STORED name contains the spoken
         one ("masoor" finding "Lentils (Masoor)")

    Step 4 deliberately does not work the other way around. "Brown Sugar"
    contains "Sugar", but they are different products, and merging them
    would silently corrupt both stock counts.
    """
    if not name or not name.strip():
        return None

    query = name.strip()

    item = db.query(model).filter(model.name.ilike(query)).first()
    if item:
        return item

    all_rows = db.query(model).all()
    if not all_rows:
        return None

    q_squash = _squash(query)
    for candidate in all_rows:
        if _squash(candidate.name) == q_squash:
            log.info("match: %r -> %r (spelling)", query, candidate.name)
            return candidate

    q_norm = _norm(query)
    by_norm = {_norm(c.name): c for c in all_rows}
    close = difflib.get_close_matches(q_norm, list(by_norm.keys()),
                                      n=1, cutoff=cutoff)
    if close:
        candidate = by_norm[close[0]]
        log.info("match: %r -> %r (close)", query, candidate.name)
        return candidate

    if allow_partial and len(q_norm) >= 3:
        contains = [c for c in all_rows if q_norm in _norm(c.name)]
        if len(contains) == 1:
            log.info("match: %r -> %r (partial)", query, contains[0].name)
            return contains[0]
        if len(contains) > 1:
            log.info("match: %r is ambiguous across %s rows",
                     query, len(contains))

    return None


def find_item(db, name, allow_partial=False, cutoff=FUZZY_CUTOFF_VOICE):
    return find_by_name(db, models.Item, name, allow_partial, cutoff)


def find_customer(db, name, allow_partial=True):
    return find_by_name(db, models.Customer, name, allow_partial,
                        FUZZY_CUTOFF_CUSTOMER)


def customer_balance(db, customer_id):
    """Positive = customer owes the shop. Negative = the shop owes them."""
    rows = db.query(models.KhataEntry).filter(
        models.KhataEntry.customer_id == customer_id).all()
    udhaar = sum(r.amount or 0.0 for r in rows if r.type == 'udhaar')
    jama = sum(r.amount or 0.0 for r in rows if r.type == 'jama')
    return round(udhaar - jama, 2)


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


def to_pkt(dt):
    return dt.replace(tzinfo=datetime.timezone.utc).astimezone(PKT)


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
    NOT Rs 1450 per gram. Multiplying gives Rs 1,377,500.

    The tell is that RATE and AMOUNT are the same number.
    """
    if qty > 1 and line_amount > 0 and abs(line_amount - price) < 0.01:
        packed_name = item_name
        size = f"{qty:g}{unit}".replace(" ", "")
        if size.lower() not in item_name.lower().replace(" ", ""):
            packed_name = f"{item_name} {qty:g}{unit}"
        log.info("scan-bill: %r priced per pack, storing 1 pack at %s",
                 item_name, price)
        return 1.0, "pack", price, packed_name

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
        Ignore any TOTAL, SUBTOTAL, CASH, CHANGE, TAX or DISCOUNT rows.
        Translate names to English (e.g., Namak to Salt, Chini to Sugar).
        Use the most common English spelling: "Chilli", "Yogurt", "Lentils".

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

            qty, unit, price, item_name = normalize_pack_pricing(
                item_name, qty, unit, price, line_amount)

            # Strict matching: a wrong merge on a scan is worse than a
            # duplicate row, because it silently inflates someone's stock.
            db_item = find_item(db, item_name, allow_partial=False,
                                cutoff=FUZZY_CUTOFF_SCAN)

            if not db_item:
                db_item = models.Item(name=item_name, quantity=0.0, unit=unit)
                db.add(db_item)
                db.flush()

            if price > 0:
                db_item.cost_price = price
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

            results_summary.append(f"Added {qty:g} {unit} of {db_item.name}")
            items_for_flutter.append({
                "name": db_item.name,
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
VOICE_WRITE_ACTIONS = ("STOCK_IN", "STOCK_OUT", "KHATA_UDHAAR", "KHATA_JAMA")


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

    KHATA (customer credit ledger) — these ALWAYS name a person:
    - 'X ko ... udhaar', 'X ko ... dediya', 'X ke khate mein likho'
        -> "KHATA_UDHAAR"  (customer took goods or cash on credit)
    - 'X ne ... rupay diye', 'X ne payment ki', 'X se ... mila'
        -> "KHATA_JAMA"    (customer paid money back)
    - 'X ka hisab', 'X kitna deta hai', 'X ka balance'
        -> "KHATA_BALANCE"

    For khata commands put the PERSON'S NAME in "customer".
    If goods were given, also fill "item", "qty" and "unit" — leave "amount"
    as 0 and the server will price it from stock.
    If only money was involved, fill "amount" and leave "item" empty.

    Examples:
    "Ali ko 5 kilo chini udhaar" ->
      {{"action":"KHATA_UDHAAR","customer":"Ali","item":"Sugar","qty":5,
        "unit":"kg","amount":0,"price":0,
        "voice_response":"Ali ke khate mein 5 kilo cheeni likh di."}}
    "Ali ne 1000 rupay diye" ->
      {{"action":"KHATA_JAMA","customer":"Ali","item":"","qty":0,
        "unit":"","amount":1000,"price":0,
        "voice_response":"Ali se 1000 rupay wasool huye."}}

    Return ONLY a JSON ARRAY, even when there is just one command:
    [
      {{
        "action": "STOCK_IN / STOCK_OUT / QUERY / SUMMARY / KHATA_UDHAAR / KHATA_JAMA / KHATA_BALANCE",
        "item": "Standard English Name",
        "customer": "Person name, or empty",
        "qty": 0.0,
        "unit": "kg / pcs / pack",
        "price": 0.0,
        "amount": 0.0,
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
            customer_name = (a.get("customer") or "").strip()
            qty = to_float(a.get("qty"))

            if action == "QUERY":
                db_item = find_item(db, item_name, allow_partial=True)
                if db_item:
                    a["item"] = db_item.name
                    a["voice_response"] = (
                        f"{db_item.name} ka stock {db_item.quantity:g} "
                        f"{db_item.unit} bacha hai.")
                else:
                    a["voice_response"] = (
                        f"Maaf kijie, {item_name} record mein nahi mila.")

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

            elif action == "KHATA_BALANCE":
                cust = find_customer(db, customer_name)
                if not cust:
                    a["voice_response"] = (
                        f"{customer_name} ka koi khata nahi hai.")
                else:
                    a["customer"] = cust.name
                    bal = customer_balance(db, cust.id)
                    a["balance"] = bal
                    if bal > 0:
                        a["voice_response"] = (
                            f"{cust.name} ne {bal:,.0f} rupay dene hain.")
                    elif bal < 0:
                        a["voice_response"] = (
                            f"Aap ne {cust.name} ko {abs(bal):,.0f} "
                            f"rupay dene hain.")
                    else:
                        a["voice_response"] = f"{cust.name} ka hisab saaf hai."

            elif action in ("KHATA_UDHAAR", "KHATA_JAMA"):
                if not customer_name:
                    a["warning"] = "Kis ka khata? Naam nahi samjha."
                    continue

                cust = find_customer(db, customer_name)
                if cust:
                    if cust.name != customer_name:
                        a["matched_from"] = customer_name
                    a["customer"] = cust.name
                    a["current_balance"] = customer_balance(db, cust.id)
                else:
                    # Not an error — a new khata gets opened on confirm.
                    # But say so, because a mis-heard name should not
                    # silently create a second account for the same person.
                    a["new_customer"] = True
                    a["current_balance"] = 0.0

                if action == "KHATA_UDHAAR" and item_name:
                    db_item = find_item(db, item_name, allow_partial=True)
                    if not db_item:
                        a["warning"] = f"{item_name} stock mein nahi hai."
                    elif db_item.quantity < qty:
                        a["warning"] = (
                            f"Sirf {db_item.quantity:g} {db_item.unit} "
                            f"{db_item.name} bacha hai.")
                    else:
                        a["item"] = db_item.name
                        rate = (to_float(a.get("price"))
                                or float(db_item.sale_price or 0.0))
                        a["unit_price"] = rate
                        a["amount"] = round(qty * rate, 2)
                        if rate <= 0:
                            a["warning"] = (
                                f"{db_item.name} ka rate set nahi hai.")
                else:
                    a["amount"] = to_float(a.get("amount"))
                    if a["amount"] <= 0:
                        a["warning"] = "Kitne rupay? Amount nahi samjha."

            elif action in ("STOCK_IN", "STOCK_OUT"):
                db_item = find_item(db, item_name, allow_partial=True)

                # THE KEY STEP: rewrite the name to the one actually stored,
                # so the confirm phase looks up an exact match.
                if db_item and db_item.name != item_name:
                    log.info("voice: rewriting %r as %r",
                             item_name, db_item.name)
                    a["matched_from"] = item_name
                    a["item"] = db_item.name

                if action == "STOCK_OUT":
                    if not db_item:
                        a["warning"] = f"{item_name} stock mein nahi hai."
                    elif db_item.quantity < qty:
                        a["warning"] = (
                            f"Sirf {db_item.quantity:g} {db_item.unit} "
                            f"{db_item.name} bacha hai.")
                    else:
                        rate = (to_float(a.get("price"))
                                or float(db_item.sale_price or 0.0))
                        a["unit_price"] = rate
                        a["line_total"] = round(qty * rate, 2)

        needs_confirm = any(
            a.get("action") in VOICE_WRITE_ACTIONS for a in actions)
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
    """Commits confirmed write actions. All-or-nothing."""
    results = []
    sale_total = 0.0
    try:
        for a in actions:
            action = a.get("action")
            if action not in VOICE_WRITE_ACTIONS:
                continue

            if action in ("KHATA_UDHAAR", "KHATA_JAMA"):
                line, credited = _commit_khata(a, db)
                results.append(line)
                sale_total += credited
                continue

            item_name = (a.get("item") or "").strip()
            qty = to_float(a.get("qty"))
            if qty <= 0:
                raise ValueError(f"Invalid quantity for {item_name}")

            db_item = find_item(db, item_name, allow_partial=True)

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
                        f"Sirf {db_item.quantity:g} {db_item.unit} "
                        f"{db_item.name} bacha hai")
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

            if action == "STOCK_OUT":
                results.append(
                    f"Sold {qty:g} {db_item.unit} {db_item.name} — "
                    f"Rs {line_total:,.0f} "
                    f"({db_item.quantity:g} {db_item.unit} left)")
            else:
                results.append(
                    f"Added {qty:g} {db_item.unit} {db_item.name} "
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


def _commit_khata(a, db):
    """Writes one khata entry. Returns (summary line, revenue recognised).

    Goods given on credit are a SALE — stock leaves and the shop has earned
    the money, it just has not been paid yet. So this writes both a
    StockTransaction and a KhataEntry. Cash repayments touch neither stock
    nor revenue; they only move the balance.
    """
    action = a.get("action")
    customer_name = (a.get("customer") or "").strip()
    if not customer_name:
        raise ValueError("Customer ka naam nahi mila")

    cust = find_customer(db, customer_name)
    if not cust:
        cust = models.Customer(name=customer_name)
        db.add(cust)
        db.flush()
        log.info("khata: opened new account for %r", customer_name)

    item_name = (a.get("item") or "").strip()
    qty = to_float(a.get("qty"))
    amount = to_float(a.get("amount"))
    revenue = 0.0

    # --- money only ---
    if action == "KHATA_JAMA" or not item_name:
        if amount <= 0:
            raise ValueError("Amount zero hai")

        db.add(models.KhataEntry(
            customer_id=cust.id,
            type='jama' if action == "KHATA_JAMA" else 'udhaar',
            amount=amount,
            note=a.get("note"),
        ))
        bal = customer_balance(db, cust.id)
        verb = "paid" if action == "KHATA_JAMA" else "took"
        return (f"{cust.name} {verb} Rs {amount:,.0f} "
                f"(balance Rs {bal:,.0f})"), revenue

    # --- goods on credit ---
    db_item = find_item(db, item_name, allow_partial=True)
    if not db_item:
        raise ValueError(f"{item_name} stock mein nahi hai")
    if db_item.quantity < qty:
        raise ValueError(f"Sirf {db_item.quantity:g} {db_item.unit} "
                         f"{db_item.name} bacha hai")

    rate = (to_float(a.get("unit_price")) or to_float(a.get("price"))
            or float(db_item.sale_price or 0.0))
    if rate <= 0:
        raise ValueError(f"{db_item.name} ka rate set nahi hai")

    line_total = round(qty * rate, 2)

    db_item.quantity -= qty
    db.add(models.StockTransaction(
        item_id=db_item.id,
        type="out",
        quantity=qty,
        unit_price=rate,
        total_amount=line_total,
    ))
    db.add(models.KhataEntry(
        customer_id=cust.id,
        type='udhaar',
        amount=line_total,
        item_id=db_item.id,
        quantity=qty,
        unit_price=rate,
        note=f"{qty:g} {db_item.unit} {db_item.name}",
    ))
    revenue = line_total

    bal = customer_balance(db, cust.id)
    return (f"{cust.name} took {qty:g} {db_item.unit} {db_item.name} "
            f"on credit — Rs {line_total:,.0f} (balance Rs {bal:,.0f})"), revenue


# -------------------------------------------------------------------------
# 4. KHATA LEDGER
# -------------------------------------------------------------------------
class CustomerIn(BaseModel):
    name: str
    phone: str | None = None


class EntryIn(BaseModel):
    type: str               # 'udhaar' or 'jama'
    amount: float
    note: str | None = None


@app.get("/khata/customers")
def list_customers(db: Session = Depends(get_db)):
    """Everyone with an account, and what each one owes.

    Sorted by who owes most, because that is the list a shopkeeper
    actually wants to look at.
    """
    customers = db.query(models.Customer).all()
    entries = db.query(models.KhataEntry).all()

    totals = {}
    last_seen = {}
    for e in entries:
        bucket = totals.setdefault(e.customer_id, {"udhaar": 0.0, "jama": 0.0})
        bucket[e.type] = bucket.get(e.type, 0.0) + (e.amount or 0.0)
        if e.timestamp and (e.customer_id not in last_seen
                            or e.timestamp > last_seen[e.customer_id]):
            last_seen[e.customer_id] = e.timestamp

    out = []
    for c in customers:
        t = totals.get(c.id, {"udhaar": 0.0, "jama": 0.0})
        balance = round(t.get("udhaar", 0.0) - t.get("jama", 0.0), 2)
        seen = last_seen.get(c.id)
        out.append({
            "id": c.id,
            "name": c.name,
            "phone": c.phone,
            "balance": balance,
            "total_udhaar": round(t.get("udhaar", 0.0), 2),
            "total_jama": round(t.get("jama", 0.0), 2),
            "last_activity": to_pkt(seen).isoformat() if seen else None,
        })

    out.sort(key=lambda x: x["balance"], reverse=True)

    return {
        "customers": out,
        "total_receivable": round(
            sum(c["balance"] for c in out if c["balance"] > 0), 2),
        "total_payable": round(
            sum(-c["balance"] for c in out if c["balance"] < 0), 2),
    }


@app.post("/khata/customers")
def add_customer(body: CustomerIn, db: Session = Depends(get_db)):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")

    existing = find_customer(db, name, allow_partial=False)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"{existing.name} already has a khata")

    cust = models.Customer(name=name, phone=body.phone)
    db.add(cust)
    db.commit()
    return {"status": "success", "id": cust.id, "name": cust.name,
            "phone": cust.phone, "balance": 0.0}


@app.get("/khata/customers/{customer_id}")
def customer_detail(customer_id: int, db: Session = Depends(get_db)):
    """One customer's full ledger, newest entry first, with a running
    balance so the shopkeeper can see how it got to where it is."""
    cust = db.query(models.Customer).filter(
        models.Customer.id == customer_id).first()
    if not cust:
        raise HTTPException(status_code=404, detail="Customer not found")

    rows = (db.query(models.KhataEntry)
            .filter(models.KhataEntry.customer_id == customer_id)
            .order_by(models.KhataEntry.timestamp).all())

    running = 0.0
    entries = []
    for r in rows:
        amount = r.amount or 0.0
        running += amount if r.type == 'udhaar' else -amount
        entries.append({
            "id": r.id,
            "type": r.type,
            "amount": round(amount, 2),
            "note": r.note,
            "quantity": float(r.quantity) if r.quantity else None,
            "unit_price": float(r.unit_price) if r.unit_price else None,
            "balance_after": round(running, 2),
            "timestamp": to_pkt(r.timestamp).isoformat() if r.timestamp else None,
        })

    entries.reverse()

    return {
        "id": cust.id,
        "name": cust.name,
        "phone": cust.phone,
        "balance": round(running, 2),
        "entry_count": len(entries),
        "entries": entries,
    }


@app.post("/khata/customers/{customer_id}/entries")
def add_entry(customer_id: int, body: EntryIn, db: Session = Depends(get_db)):
    """Manual ledger entry — cash lent or repaid, no stock involved."""
    cust = db.query(models.Customer).filter(
        models.Customer.id == customer_id).first()
    if not cust:
        raise HTTPException(status_code=404, detail="Customer not found")

    entry_type = (body.type or "").strip().lower()
    if entry_type not in ("udhaar", "jama"):
        raise HTTPException(status_code=400,
                            detail="type must be 'udhaar' or 'jama'")
    if body.amount <= 0:
        raise HTTPException(status_code=400,
                            detail="Amount must be greater than zero")

    db.add(models.KhataEntry(
        customer_id=cust.id,
        type=entry_type,
        amount=body.amount,
        note=body.note,
    ))
    db.commit()

    return {"status": "success", "customer": cust.name,
            "balance": customer_balance(db, cust.id)}


@app.delete("/khata/entries/{entry_id}")
def delete_entry(entry_id: int, db: Session = Depends(get_db)):
    """Removes a ledger line. Does NOT restore stock — if goods went out on
    a mistaken entry, correct the stock separately so the two decisions stay
    visible rather than one silently undoing the other."""
    entry = db.query(models.KhataEntry).filter(
        models.KhataEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    customer_id = entry.customer_id
    had_stock = entry.item_id is not None
    db.delete(entry)
    db.commit()

    return {
        "status": "success",
        "balance": customer_balance(db, customer_id),
        "stock_unchanged": had_stock,
    }


# -------------------------------------------------------------------------
# 5. REPORTS
# -------------------------------------------------------------------------
@app.get("/reports/daily")
def daily_report(
    date: str | None = Query(None, description="YYYY-MM-DD, defaults to today"),
    db: Session = Depends(get_db),
):
    """Day-end record.

    Note that total_sale and cash_in_hand are different numbers once credit
    exists. Goods sold on udhaar count as revenue the moment they leave the
    shop, but no cash arrived. A shopkeeper looking at a Rs 5,000 sale total
    with Rs 2,000 in the drawer needs to see why.
    """
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
                "time": to_pkt(txn.timestamp).strftime("%H:%M"),
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

        # Khata movement for the same day.
        khata_rows = (
            db.query(models.KhataEntry, models.Customer)
            .join(models.Customer,
                  models.Customer.id == models.KhataEntry.customer_id)
            .filter(models.KhataEntry.timestamp >= start,
                    models.KhataEntry.timestamp < end)
            .order_by(models.KhataEntry.timestamp).all())

        credit_given = payments_received = 0.0
        khata_lines = []
        for entry, cust in khata_rows:
            amount = float(entry.amount or 0.0)
            if entry.type == 'udhaar':
                credit_given += amount
            else:
                payments_received += amount
            khata_lines.append({
                "time": to_pkt(entry.timestamp).strftime("%H:%M"),
                "customer": cust.name,
                "type": entry.type,
                "amount": amount,
                "note": entry.note,
            })

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
            # Money actually collected: sales that were not on credit, plus
            # any repayments that came in today.
            "credit_given": round(credit_given, 2),
            "payments_received": round(payments_received, 2),
            "cash_in_hand": round(revenue - credit_given + payments_received, 2),
            "items_sold": sorted(by_item.values(),
                                 key=lambda x: x["amount"], reverse=True),
            "sales": sales,
            "purchases": purchases,
            "khata": khata_lines,
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
# 6. UTILITY ENDPOINTS
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


@app.get("/items/{item_id}/transactions")
def item_transactions(item_id: int, limit: int = Query(50, ge=1, le=500),
                      db: Session = Depends(get_db)):
    """Movement history for one item, newest first."""
    item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    rows = (db.query(models.StockTransaction)
            .filter(models.StockTransaction.item_id == item_id)
            .order_by(models.StockTransaction.timestamp.desc())
            .limit(limit).all())

    return {
        "item": item.name,
        "unit": item.unit,
        "transactions": [{
            "id": r.id,
            "type": r.type,
            "quantity": float(r.quantity or 0.0),
            "unit_price": float(r.unit_price or 0.0),
            "amount": float(r.total_amount or 0.0),
            "timestamp": to_pkt(r.timestamp).isoformat(),
        } for r in rows],
    }


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
    """Correct an item the scanner got wrong — wrong unit, quantity, price."""
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
import os
import json
import datetime
import re
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

# Local Imports
import models
from database import engine, get_db, Base

# 1. INITIALIZATION
load_dotenv()
Base.metadata.create_all(bind=engine)
app = FastAPI(title="AwazKhata AI Backend 2026")

# Setup Gemini 2026 Client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class VoiceInput(BaseModel):
    transcript: str

@app.get("/")
def home():
    return {"status": "Online", "message": "AwazKhata API is active"}

# 2. MAIN VOICE PROCESSOR
@app.post("/voice/process")
async def process_voice(data: VoiceInput, db: Session = Depends(get_db)):
    # Define the prompt for the AI
    prompt = f"""
    Acting as AwazKhata AI. Convert this shopkeeper's command into JSON: "{data.transcript}"
    
    Actions: STOCK_IN, STOCK_OUT, QUERY, SUMMARY.
    Return ONLY this JSON format:
    {{
      "action": "ACTION_TYPE",
      "item": "English Name",
      "qty": 0.0,
      "unit": "kg/pcs",
      "voice_response": "Short Urdu response"
    }}
    """
    
    try:
        # Call Gemini 3.5 Flash reasoning
        response = client.models.generate_content(
            model="gemini-3.5-flash", 
            contents=prompt,
            config={
                # 3.5 Flash supports native JSON output, 
                # which stops the AI from adding markdown or extra text.
                'response_mime_type': 'application/json',
            }
        )
        
        # Clean AI response (handle cases where AI adds markdown)
        raw_text = response.text
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        
        if match:
            intent = json.loads(response.text)
        else:
            raise ValueError("AI response did not contain valid JSON")

        action = intent.get("action")
        item_name = intent.get("item", "").strip()
        qty = float(intent.get("qty", 0.0))

        # --- DATABASE LOGIC ---
        
        # Case 1: QUERY (Check Stock)
        if action == "QUERY":
            item = db.query(models.Item).filter(models.Item.name.ilike(item_name)).first()
            if item:
                intent["voice_response"] = f"{item.name} ka stock {item.quantity} {item.unit} bacha hai."
            else:
                intent["voice_response"] = f"Maaf kijie, {item_name} record mein nahi mila."

        # Case 2: STOCK UPDATES
        elif action in ["STOCK_IN", "STOCK_OUT"]:
            item = db.query(models.Item).filter(models.Item.name.ilike(item_name)).first()
            
            if not item and action == "STOCK_IN":
                item = models.Item(name=item_name, quantity=0.0, unit=intent.get("unit", "kg"))
                db.add(item)
                db.flush()

            if item:
                if action == "STOCK_IN":
                    item.quantity += qty
                else:
                    item.quantity -= qty
                
                # Record the transaction
                new_tx = models.StockTransaction(
                    item_id=item.id,
                    type=action.lower().replace("stock_", ""),
                    quantity=qty
                )
                db.add(new_tx)
                db.commit()
            else:
                intent["voice_response"] = f"Pehle {item_name} ko add karein."

        # Case 3: SUMMARY
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
        print(f"❌ ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI/DB Error: {str(e)}")

@app.get("/health")
def status():
    return {"status": "ok", "api_key_loaded": api_key is not None}
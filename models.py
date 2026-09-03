from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from database import Base
import datetime

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    quantity = Column(Float, default=0.0)
    unit = Column(String, default="kg")
    min_stock = Column(Float, default=5.0)

class StockTransaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("items.id"))
    type = Column(String)  # 'in' or 'out'
    quantity = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
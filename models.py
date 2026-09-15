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

    # What the shopkeeper SELLS this for (per unit).
    sale_price = Column(Float, default=0.0)

    # What the shopkeeper PAID for it (per unit), read off purchase bills.
    # Kept separate so profit can be worked out later.
    cost_price = Column(Float, default=0.0)


class StockTransaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("items.id"))
    type = Column(String)  # 'in' or 'out'
    quantity = Column(Float)

    # Price captured AT THE MOMENT of the transaction. Storing it here
    # rather than reading Item.sale_price later matters: if the price
    # changes next week, last week's sales must not change with it.
    unit_price = Column(Float, default=0.0)
    total_amount = Column(Float, default=0.0)

    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
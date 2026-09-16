from sqlalchemy import (Column, Integer, String, Float, DateTime, ForeignKey,
                        Index)
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


class Customer(Base):
    """Anyone with a running account at the shop.

    Can be a buyer who takes goods on credit, or a supplier the shop owes —
    the balance simply runs the other way. One table covers both.
    """
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    phone = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class KhataEntry(Base):
    """One line in a customer's ledger.

    type 'udhaar' increases what the customer owes (they took goods or cash).
    type 'jama'   decreases it (they paid, or the shop owes them).

    Balance = sum(udhaar) - sum(jama).
      positive -> the customer owes the shop
      negative -> the shop owes the customer

    When an entry came from goods leaving the shop, item_id / quantity /
    unit_price are filled in and a matching StockTransaction exists. When it
    was plain cash, they stay null.
    """
    __tablename__ = "khata_entries"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), index=True)
    type = Column(String)  # 'udhaar' or 'jama'
    amount = Column(Float, default=0.0)
    note = Column(String, nullable=True)

    item_id = Column(Integer, ForeignKey("items.id"), nullable=True)
    quantity = Column(Float, nullable=True)
    unit_price = Column(Float, nullable=True)

    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)


Index("ix_khata_customer_time", KhataEntry.customer_id, KhataEntry.timestamp)
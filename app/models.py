"""
models.py
database schemas for Kickstarter app
"""
# *** IMPORTS ***
from flask_sqlalchemy import SQLAlchemy


# Create a DB Object
DB = SQLAlchemy()


# *** Define Models/Tables ***
class Kickstarter(DB.Model):
    """
    Defines the "Kickstarters" table with SQLAlchemy
    """
    id = DB.Column(DB.BigInteger, primary_key=True)
    name = DB.Column(DB.String, nullable=False)
    blurb = DB.Column(DB.String, nullable=True)
    goal = DB.Column(DB.Float, nullable=True)
    campaign_duration = DB.Column(DB.Float, nullable=True)
    current_currency = DB.Column(DB.String, nullable=True)
    fx_rate = DB.Column(DB.Float, nullable=True)
    static_usd_rate = DB.Column(DB.Float, nullable=True)
    outcome = DB.Column(DB.Boolean, nullable=True)
    days_to_success = DB.Column(DB.Float, nullable=True)
    city = DB.Column(DB.String, nullable=True)
    state = DB.Column(DB.String, nullable=True)
    country = DB.Column(DB.String, nullable=True)
    category = DB.Column(DB.String, nullable=True)
    subcategory = DB.Column(DB.String, nullable=True)
    location = DB.Column(DB.String, nullable=True)
    latitude = DB.Column(DB.Float, nullable=True)
    longitude = DB.Column(DB.Float, nullable=True)

    def __repr__(self):
        return f'Kickstarter: name - {self.name}'

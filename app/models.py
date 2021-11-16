"""
models.py
database schemas for Kickstarter app
"""
# *** IMPORTS ***
from flask_sqlalchemy import SQLAlchemy


# Create a DB Object
DB = SQLAlchemy()


# *** Define Models/Tables ***
class Kickstarters(DB.Model):
    """
    Defines the "Kickstarters" table with SQLAlchemy
    """
    id = DB.Column(DB.BigInteger, primary_key=True)
    name = DB.Column(DB.String)
    blurb = DB.Column(DB.String)
    goal = DB.Column(DB.Float)
    campaign_duration = DB.Column(DB.Float)
    current_currency = DB.Column(DB.String)
    fx_rate = DB.Column(DB.Float)
    static_usd_rate = DB.Column(DB.Float)
    outcome = DB.Column(DB.Boolean)
    days_to_success = DB.Column(DB.Float)
    city = DB.Column(DB.String)
    state = DB.Column(DB.String)
    country = DB.Column(DB.String)
    category = DB.Column(DB.String)
    subcategory = DB.Column(DB.String)
    location = DB.Column(DB.String)
    latitude = DB.Column(DB.Float)
    longitude = DB.Column(DB.Float)
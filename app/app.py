"""
app.py - main app for the Kickstarter evaluator tool
"""

# *** IMPORTS ***
from .models import DB, Kickstarter, engine
from os import getenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import pandas as pd


def create_app():
    app = Flask(__name__)

    # Give our APP access to our database
    DB.init_app(app)

    @app.route('/')
    def hello_world():  # put application's code here
        return 'Hello World!'

    @app.route('/db')
    def load_db():
        """
        Empty and repopulate the database with tables defined in models.py
        :return:
        """
        DB.drop_all()
        DB.create_all()
        new_ks = Kickstarter(id=1, name='foop')
        DB.session.add(new_ks)
        ks = Kickstarter.query.all()
        ks_string = str(ks[0])
        DB.session.commit()
        df = pd.read_sql('select * from test limit 5;',
                         con=engine)
        # DB.table('Kickstarter')
        return 'Database updated!' + ks_string  + str(df.head())

    return app


if __name__ == '__main__':
    app = create_app()
    app.run()

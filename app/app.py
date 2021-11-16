"""
app.py - main app for the Kickstarter evaluator tool
"""

# *** IMPORTS ***
from .models import DB
from os import getenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


def create_app():
    app = Flask(__name__)

    @app.route('/')
    def hello_world():  # put application's code here
        return 'Hello World!' + getenv('DATABASE_URL')


    @app.route('/db')
    def load_db():
        """

        :return:
        """
        DB.drop_all()
        DB.create_all()
        return 'Database updated!'

    return app


if __name__ == '__main__':
    app = create_app()
    app.run()

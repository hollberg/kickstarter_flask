"""
app.py - main app for the Kickstarter evaluator tool
"""

# *** IMPORTS ***
from os import getenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


def create_app():
    app = Flask(__name__)

    @app.route('/')
    def hello_world():  # put application's code here
        return 'Hello World!' + getenv('DATABASE_URL')
    return app


    @app.route('/db')
    def load_db():
        """

        :return:
        """




if __name__ == '__main__':
    app = create_app()
    app.run()

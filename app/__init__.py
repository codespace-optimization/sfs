from flask import Flask
from .routes import routes

def create_app():
    app = Flask(__name__)
    app.register_blueprint(routes)
    return app

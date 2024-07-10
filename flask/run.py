import ngrok
from flaskr import create_app
from flask_cors import CORS
from flask_sslify import SSLify

app = create_app()
sslify = SSLify(app)
CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5000"}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

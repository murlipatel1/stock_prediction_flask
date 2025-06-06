from flask import Flask, jsonify
from flask_cors import CORS
from routes import api_bp
import logging
import os

def create_app():
    app = Flask(__name__)
    
    # Enable CORS with more specific configuration for Power BI integration
    cors_config = {
        "origins": ["*"],  # Allow all origins, or restrict to specific ones
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "max_age": 3600  # Cache preflight requests for 1 hour
    }
    CORS(app, resources={r"/api/*": cors_config})
    
    # Configure logging
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/api.log'),
            logging.StreamHandler()
        ]
    )
    
    # Register blueprint
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "service": "Indian Stock Predictor API"})
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", debug=True)

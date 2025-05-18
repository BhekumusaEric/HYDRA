from flask import Flask, request, jsonify
import os
import time
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simulated user database (in a real app, this would be in a database)
USERS = {
    'admin': 'secret123',
    'user1': 'password123',
    'guest': 'guest'
}

# Simulated session tracking
SESSIONS = {}

# Intentional vulnerabilities for the Red Agent to discover

# Vulnerability 1: No rate limiting on login attempts (brute force vulnerability)
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Log the login attempt
    logger.info(f"Login attempt for user: {username}")
    
    if not username or not password:
        return jsonify({"status": "error", "message": "Missing username or password"}), 400
    
    # Check credentials (vulnerable to brute force)
    if username in USERS and USERS[username] == password:
        # Create a simple session token
        session_token = f"{username}_{int(time.time())}"
        SESSIONS[session_token] = username
        
        return jsonify({"status": "success", "message": "Login successful", "token": session_token})
    
    return jsonify({"status": "error", "message": "Invalid credentials"}), 401

# Vulnerability 2: No input validation (SQL injection simulation)
@app.route('/user_info', methods=['GET'])
def user_info():
    user_id = request.args.get('id')
    
    # Log the request
    logger.info(f"User info requested for ID: {user_id}")
    
    # Simulate SQL injection vulnerability
    if "'" in user_id or ";" in user_id:
        # In a real app, this would be a successful SQL injection
        # Here we just simulate it by returning all users
        return jsonify({"status": "success", "users": list(USERS.keys())}), 200
    
    # Normal response for a valid user ID
    if user_id in ['1', '2', '3']:
        user_map = {'1': 'admin', '2': 'user1', '3': 'guest'}
        return jsonify({"status": "success", "user": user_map[user_id]}), 200
    
    return jsonify({"status": "error", "message": "User not found"}), 404

# Vulnerability 3: Insecure direct object reference
@app.route('/download', methods=['GET'])
def download_file():
    filename = request.args.get('file')
    
    # Log the download request
    logger.info(f"File download requested: {filename}")
    
    # No path validation (vulnerable to path traversal)
    if filename:
        # In a real app, this would serve the file
        # Here we just simulate the vulnerability
        if '../' in filename or '/' in filename:
            return jsonify({"status": "success", "message": f"Simulated path traversal: {filename}"}), 200
        
        return jsonify({"status": "success", "message": f"File would be: {filename}"}), 200
    
    return jsonify({"status": "error", "message": "No file specified"}), 400

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "up"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

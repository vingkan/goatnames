"""
Server to produce on-demand XG predictions.
"""

import sys
sys.path.append("./")

from flask import (
    Flask,
    jsonify,
    request,
    Response,
    send_file,
    render_template
)
from flask_cors import CORS
from clues import give_clues


# Initialize Flask app and enable CORS.
app = Flask(__name__)
allow_list = [
    "http://localhost:4000"
]
cors = CORS(app, resource={"/*": {"origins": allow_list}})


@app.route("/hello")
def hello():
    """
    Basic hello world route to check if server is running.
    """
    return "Welcome to Codenames."


@app.route("/")
def index():
    """
    Serve home page.
    """
    return render_template("index.html")


@app.route("/api/cluegiver", methods=["POST"])
def api_clue_giver():
    """
    Give clues.
    """
    args = request.json
    res = give_clues(**args)
    return jsonify(res)


# Start the server on the default host.
if __name__ == "__main__":
    print("Starting server...")
    app.run(host="0.0.0.0", port=4000)

"""
Server to produce on-demand XG predictions.
"""

import sys
sys.path.append("./")

from decouple import config
from flask import (
    Flask,
    jsonify,
    request,
    Response,
    send_file,
    render_template
)
from flask_cors import CORS
from scraper import get_board_from_horsepaste
from clues import give_clues


# Initialize Flask app and enable CORS.
app = Flask(__name__)
allow_list = config("ALLOW").strip().split(",")
cors = CORS(app, resource={"/*": {"origins": allow_list}})


@app.route("/hello")
def hello():
    """
    Basic hello world route to check if server is running.
    """
    return "Welcome to Codenames."


@app.route("/")
def page_index():
    """
    Serve home page.
    """
    return render_template("index.html")


@app.route("/board")
def page_board():
    """
    Serve clue-giver page that accepts raw board state.
    """
    return render_template("board.html")


@app.route("/horsepaste")
def page_horsepaste():
    """
    Serve clue-giver page that accepts Horsepaste URL.
    """
    return render_template("horsepaste.html")


@app.route("/api/cluegiver/board", methods=["POST"])
def api_cluegiver():
    """
    Give clues for raw board state.
    """
    args = request.json
    res = give_clues(**args, hint=True)
    return jsonify(res)


@app.route("/api/cluegiver/horsepaste", methods=["POST"])
def api_cluegiver_horsepaste():
    """
    Give clues for horsepaste URL.
    """
    args = request.json
    url = args["horsepaste"]
    board, flipped = get_board_from_horsepaste(url, config("CHROME"))
    args["board"] = board
    args["flipped"] = flipped
    del args["horsepaste"]
    res = give_clues(**args)
    return jsonify(res)


# Start the server on the default host.
if __name__ == "__main__":
    print("Starting server...")
    app.run(host="0.0.0.0", port=int(config("PORT")))

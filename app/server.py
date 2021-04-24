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
from gensim.models import KeyedVectors
from scraper import get_board_from_horsepaste
from giver import (
    parse_cards_from_board,
    ClueGiver,
    NClosestClueGiver,
    make_closest_distance_fn,
    make_most_similar_fn,
    CarefulClueGiver
)


# Initialize Flask app and enable CORS.
app = Flask(__name__)
allow_list = config("ALLOW").strip().split(",")
cors = CORS(app, resource={"/*": {"origins": allow_list}})

# Load models and stop words list
model = KeyedVectors.load("models/glove.6B.100d.kv")
with open("nltk_stop_words.txt") as file:
    stop_words = set(file.read().strip().split("\n"))

# Initialize clue givers
cg_closest = NClosestClueGiver(model, stop_words)
cg_closest.set_limit(100)
cg_closest.set_get_closest_fn(make_most_similar_fn(topn=500))

cg_careful = CarefulClueGiver(model, stop_words)
cg_careful.set_limit(100)


def set_state(cg, args):
    if "horsepaste" in args:
        url = args["horsepaste"]
        board, flipped = get_board_from_horsepaste(url, config("CHROME"))
    else:
        board = args["board"]
        flipped = args["flipped"]
    cg.set_cards_from_board(board, flipped)
    cg.set_team(args["team"])
    cg.set_hint_mode(args["hint"] if "hint" in args else True)
    if "count" in args:
        cg.set_count(args["count"])


def get_closest_clues(args):
    set_state(cg_closest, args)
    return cg_closest.give_clues()


def get_careful_clues(args):
    del args["count"]
    set_state(cg_careful, args)
    return cg_careful.give_clues()


get_clues_by_strategy = get_careful_clues


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
    res = get_clues_by_strategy(args)
    return jsonify(res)


@app.route("/api/cluegiver/horsepaste", methods=["POST"])
def api_cluegiver_horsepaste():
    """
    Give clues for horsepaste URL.
    """
    args = request.json
    res = get_clues_by_strategy(args)
    return jsonify(res)


# Start the server on the default host.
if __name__ == "__main__":
    print("Starting server...")
    app.run(host="0.0.0.0", port=int(config("PORT")))

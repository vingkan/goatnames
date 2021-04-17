from gensim.models import KeyedVectors
from itertools import chain, combinations
import numpy as np


# Load model
model = KeyedVectors.load("models/glove.6B.100d.kv")

# Load stop words
with open("nltk_stop_words.txt") as file:
    stop_words = set(file.read().strip().split("\n"))


def parse_cards_from_board(raw):
    """
    Returns a list of tuples (word, team, flipped).
    """
    out = []
    for line in raw.lower().strip().split("\n"):
        data = line.strip().split(", ")
        out.append((data[0], data[1], "revealed" in data[2]))
    return out


def give_clues(board="", flipped="", team="", count=2, hint=False):

    def is_eligible(word):
        is_stop = word in stop_words
        is_card = word in card_words
        is_plur_s = word[-1] == "s" and word[:-1] in card_words
        is_plur_es = word[-2:] == "es" and word[:-2] in card_words
        is_plur = is_plur_s or is_plur_es
        return not (is_stop or is_card or is_plur)


    def get_closest_clue_for_hand(hand, n=50):
        vecs = [model.get_vector(c) for c in hand]
        clue_dict = {}
        for card in hand:
            for w, s in model.most_similar(card, topn=n):
                if w not in clue_dict:
                    clue_dict[w] = {}
                clue_dict[w][card] = s
        candidates = []
        for w, sim_dict in clue_dict.items():
            if len(sim_dict) == len(hand) and is_eligible(w):
                vec_w = model.get_vector(w)
                dist_ss = 0
                for vec_c in vecs:
                    dist_c_w = np.linalg.norm(vec_c - vec_w)
                    dist_ss += np.square(dist_c_w)
                candidates.append((w, dist_ss))
        return sorted(candidates, key=lambda c: c[1], reverse=True)


    def get_most_similar_clue_for_hand(hand, n=50):
        clue_dict = {}
        for card in hand:
            for w, s in model.most_similar(card, topn=n):
                if w not in clue_dict:
                    clue_dict[w] = {}
                clue_dict[w][card] = s
        candidates = []
        for w, sim_dict in clue_dict.items():
            if len(sim_dict) == len(hand) and is_eligible(w):
                ss = np.sum([np.square(1 - s) for s in sim_dict.values()])
                score = np.sqrt(ss / len(hand))
                candidates.append((w, score))
        return sorted(candidates, key=lambda c: c[1], reverse=False)


    # Main routine starts here

    if team not in ("blue", "red"):
        return {
            "success": False,
            "display": "Team must be blue or red."
        }

    cards = parse_cards_from_board(board)
    board_cards = [c[0] for c in filter(lambda c: c[1] == team, cards)]
    card_words = set(board_cards)
    exclude_cards = [f.strip() for f in flipped.strip().split("\n")]
    my_cards = list(set(board_cards) - set(exclude_cards))

    sim_words_per_card = 500
    hand_combinations = combinations(my_cards, r=min(count, len(my_cards)))
    choices = []
    for hand in hand_combinations:
        for w, s in get_most_similar_clue_for_hand(hand, n=sim_words_per_card):
            choices.append((hand, w, s))

    top_choices = list(enumerate(sorted(choices, key=lambda p: p[2], reverse=False)))
    clues_to_return = min(len(top_choices), 100)
    lines = [
        f"Showing top {clues_to_return} clues:\n\tScoring by Word2Vec RMSE (lower is better)\n"
    ]
    for i, (hand, w, s) in top_choices[:clues_to_return]:
        hint_text = f"Hinting at {hand}\n\t" if hint else ""
        line = f"{i + 1}. {w} for {len(hand)}\n\t{hint_text}Score = {s:.3f}\n"
        lines.append(line)

    return {
        "success": True,
        "display": "\n".join(lines)
    }

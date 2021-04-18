from itertools import combinations
import numpy as np


def parse_cards_from_board(raw):
    """
    Returns a list of tuples (word, team, flipped).
    """
    out = []
    for line in raw.lower().strip().split("\n"):
        data = line.strip().split(", ")
        out.append((data[0], data[1], "revealed" in data[2]))
    return out


class ClueGiver:
    def __init__(self, model, stop_words):
        self.model = model
        self.stop_words = stop_words
        self.team = "blue"
        self.all_cards = []
        self.all_card_words = set()
        self.cards_flipped = set()
        self.clues_given = set()
        self.show_hint = False
        self.limit = 100

    def set_all_cards(self, cards):
        # cards = parse_cards_from_board(raw)
        self.all_cards = cards
        self.all_card_words = set([c[0] for c in cards])

    def set_team(self, team):
        if team not in ("blue", "red"):
            raise ValueError("Team must be blue or red.")
        self.team = team

    def add_flipped(self, card):
        self.cards_flipped.add(card)

    def add_given(self, clue):
        self.clues_given.add(clue)

    def set_cards_from_board(self, board, flipped):
        cards = parse_cards_from_board(board)
        self.all_cards = cards
        self.all_card_words = set([c[0] for c in cards])
        for card in flipped.strip().split("\n"):
            self.add_flipped(card)

    def set_hint_mode(self, show_hint):
        self.show_hint = show_hint

    def set_limit(self, limit):
        self.limit = limit

    def is_eligible(self, clue):
        is_stop = clue in self.stop_words
        is_card = clue in self.all_card_words
        is_plur_s = clue[-1] == "s" and clue[:-1] in self.all_card_words
        is_plur_es = clue[-2:] == "es" and clue[:-2] in self.all_card_words
        is_plur = is_plur_s or is_plur_es
        return not (is_stop or is_card or is_plur)

    def give_clues(self):
        pass


class NClosestClueGiver(ClueGiver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 2
        self.get_closest = lambda giver, hand: []

    def set_count(self, count):
        self.count = count

    def set_get_closest_fn(self, fn):
        self.get_closest = fn

    def give_clues(self):

        team_cards = [c[0] for c in filter(lambda c: c[1] == self.team, self.all_cards)]
        my_cards = list(set(team_cards) - set(self.cards_flipped))

        hand_combinations = combinations(my_cards, r=min(self.count, len(my_cards)))
        choices = []
        for hand in hand_combinations:
            for w, s in self.get_closest(self, hand):
                choices.append((hand, w, s))

        top_choices = list(enumerate(sorted(choices, key=lambda p: p[2], reverse=False)))
        clues_to_return = min(len(top_choices), self.limit)
        lines = [
            f"Showing top {clues_to_return} clues:\n\tScoring by Word2Vec RMSE (lower is better)\n"
        ]
        for i, (hand, w, s) in top_choices[:clues_to_return]:
            hint_text = f"Hinting at {hand}\n\t" if self.show_hint else ""
            line = f"{i + 1}. {w} for {len(hand)}\n\t{hint_text}Score = {s:.3f}\n"
            lines.append(line)

        return {
            "success": True,
            "display": "\n".join(lines)
        }


def make_closest_distance_fn(topn=500):

    def get_closest(giver, hand):
        vecs = [giver.model.get_vector(c) for c in hand]
        clue_dict = {}
        for card in hand:
            for w, s in giver.model.most_similar(card, topn=topn):
                if w not in clue_dict:
                    clue_dict[w] = {}
                clue_dict[w][card] = s
        candidates = []
        for w, sim_dict in clue_dict.items():
            if len(sim_dict) == len(hand) and giver.is_eligible(w):
                vec_w = giver.model.get_vector(w)
                dist_ss = 0
                for vec_c in vecs:
                    dist_c_w = np.linalg.norm(vec_c - vec_w)
                    dist_ss += np.square(dist_c_w)
                candidates.append((w, dist_ss))
        return sorted(candidates, key=lambda c: c[1], reverse=True)

    return get_closest


def make_most_similar_fn(topn=500):

    def get_closest(giver, hand):
        clue_dict = {}
        for card in hand:
            for w, s in giver.model.most_similar(card, topn=topn):
                if w not in clue_dict:
                    clue_dict[w] = {}
                clue_dict[w][card] = s
        candidates = []
        for w, sim_dict in clue_dict.items():
            if len(sim_dict) == len(hand) and giver.is_eligible(w):
                ss = np.sum([np.square(1 - s) for s in sim_dict.values()])
                score = np.sqrt(ss / len(hand))
                candidates.append((w, score))
        return sorted(candidates, key=lambda c: c[1], reverse=False)

    return get_closest

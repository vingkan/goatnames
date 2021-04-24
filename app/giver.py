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
        self.get_closest = lambda x: []

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


class CarefulClueGiver(ClueGiver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def give_clues(self):

        clues_to_cards = {}
        for (card, team, is_flipped) in self.all_cards:
            if is_flipped:
                continue
            for i, (clue, sim_score) in enumerate(self.model.most_similar(card, topn=500)):
                if not self.is_eligible(clue):
                    continue
                if clue not in clues_to_cards:
                    clues_to_cards[clue] = []
                sim_rank = i + 1
                clues_to_cards[clue].append((card, team, sim_score, sim_rank))

        pruned = {}
        final = {}
        for clue, cards in clues_to_cards.items():
            n_good_cards = 0
            n_bad_cards = 0
            n_neutral_cards = 0
            for (card, team, sim_score, sim_rank) in cards:
                if team == self.team:
                    n_good_cards += 1
                if team != self.team and team != "neutral":
                    n_bad_cards += 1
                if team == "neutral":
                    n_neutral_cards += 1
            if n_good_cards >= 1 and n_bad_cards == 0 and n_neutral_cards == 0:
                pruned[clue] = cards
                all_comparisons = []
                for (card, team, is_flipped) in self.all_cards:
                    if is_flipped:
                        continue
                    sim_score = self.model.similarity(clue, card)
                    all_comparisons.append((card, team, sim_score))
                final[clue] = list(sorted(all_comparisons, key=lambda c: c[2], reverse=True))

        candidates = []
        for clue, cards in final.items():
            # Number of consecutive good cards from best card that are most similar to the clue,
            # larger is better
            consecutive_good = 0
            # If the best card for clue is not good, skip this clue
            if cards[0][1] != self.team:
                continue
            first_sim_score = cards[0][2]
            prev_sim_score = first_sim_score
            for (card, team, sim_score) in cards:
                sim_delta = sim_score - first_sim_score
                if team != self.team:
                    # Absolute difference in similarity when going from best card for clue to last
                    # consecutive good card for clue, smaller is better
                    sim_reach = abs(prev_sim_score - first_sim_score)
                    # Absolute difference between similarity of last consecutive good card for clue
                    # and first bad card, larger is better
                    sim_dropoff = abs(sim_score - prev_sim_score)
                    stats = {
                        "consecutive_good": consecutive_good,
                        "sim_reach": sim_reach,
                        "sim_dropoff": sim_dropoff,
                        "worst_match": prev_sim_score,
                        "best_distractor": sim_score,
                        "multiple_matches": consecutive_good > 1,
                        "streak_impact": (consecutive_good * prev_sim_score)
                    }
                    if team == "neutral":
                        candidates.append((clue, cards, stats))
                    break
                consecutive_good += 1
                prev_sim_score = sim_score

        candidates = sorted(candidates, key=lambda c: c[2]["consecutive_good"], reverse=True)
        candidates = sorted(candidates, key=lambda c: c[2]["best_distractor"])
        candidates = sorted(candidates, key=lambda c: c[2]["worst_match"], reverse=True)
        candidates = sorted(candidates, key=lambda c: c[2]["streak_impact"], reverse=True)
        candidates = sorted(candidates, key=lambda c: c[2]["multiple_matches"], reverse=True)

        lines = []
        # Group by number of matches and summarize
        clues_per_num = {}
        for clue, cards, stats in candidates:
            if stats["consecutive_good"] not in clues_per_num:
                clues_per_num[stats["consecutive_good"]] = 0
            clues_per_num[stats["consecutive_good"]] += 1
        for num, count in sorted(clues_per_num.items(), key=lambda c: c[0]):
            lines.append(f"{count} clues for {num} words")
        # for clue, cards in sorted(pruned.items(), key=lambda c: len(c[1]), reverse=True):
        show_stats = False
        for clue, cards, stats in candidates:
            num = stats["consecutive_good"]
            lines.append(f"\n{clue} for {num}")
            if show_stats:
                stat_line = "\n".join([f"{k}: {v:.3f}" for k, v in stats.items()])
                lines.append(f"---\n{stat_line}\n---")
            first_sim_score = cards[0][2]
            for i, (card, team, sim_score) in enumerate(cards):
                if i == num:
                    lines.append("\t---")
                lines.append(f"\t{card}, {team}, {sim_score:.3f}")

        return {
            "success": True,
            "display": "\n".join(lines)
        }

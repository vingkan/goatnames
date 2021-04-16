from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from itertools import chain, combinations
import numpy as np

# Bringing back the SAUCE
model = KeyedVectors.load("models/glove.6B.100d.kv")

# Load stop words
with open("nltk_stop_words.txt") as file:
    stop_words = set(file.read().strip().split("\n"))

# Load game board
board_name = "athletics-seal"
with open(f"boards/{board_name}.txt") as file:
    cards = [tuple(line.split(", ")[:2]) for line in file.read().strip().split("\n")]

# Get spymaster cards
team = "blue"
my_cards = [c[0] for c in filter(lambda c: c[1] == team, cards)]
card_words = set(my_cards)

# Manually take away cards that have been guessed
# exclude_cards = [
    
# ]
# my_cards = list(set(my_cards) - set(exclude_cards))


def power_set(cards, m, n): 
    return chain.from_iterable(combinations(cards, t) for t in range(m, n+1))

# /shipit
def is_eligible(word):
    is_stop = word in stop_words
    is_card = word in card_words
    is_plur_s = word[-1] == "s" and word[:-1] in card_words
    is_plur_es = word[-2:] == "es" and word[:-2] in card_words
    is_plur = is_plur_s or is_plur_es
    return not (is_stop or is_card or is_plur)


def get_closest_clue(pair, n=50):
    a, b = pair
    # norm = model.similarity(a, b)
    words_a = model.most_similar(a, topn=n)
    words_b = model.most_similar(b, topn=n)
    vec_a = model.get_vector(a)
    vec_b = model.get_vector(b)
    dist_a_b = np.linalg.norm(vec_b - vec_a)
    set_a = { w: np.linalg.norm(model.get_vector(w) - vec_a) for (w, s) in words_a }
    candidates = []
    for w, s in words_b:
        if w in set_a and is_eligible(w):
            dist_a_w = set_a[w]
            dist_b_w = np.linalg.norm(model.get_vector(w) - vec_b)
            dist_ss = np.square(dist_a_w) + np.square(dist_b_w)
            dist_ss_norm = dist_ss / np.square(dist_a_b)
            candidates.append((w, dist_ss_norm))
    return list(sorted(candidates, key=lambda c: c[1], reverse=True))


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
    return list(sorted(candidates, key=lambda c: c[1], reverse=True))


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
    return list(sorted(candidates, key=lambda c: c[1], reverse=False))


print(my_cards)

top_n = 500
cards_power_set = list(power_set(my_cards, 2, 5))
common_words = [(card_set, get_most_similar_clue_for_hand(card_set, n=top_n)) for card_set in cards_power_set]

choices = []
for hand, clues in common_words:
    for w, s in clues:
        choices.append((hand, w, s))

top_choices = list(enumerate(sorted(choices, key=lambda p: p[2], reverse=False)))
for i, (hand, w, s) in top_choices[:100]:
    print(f"{i + 1}. ({hand}) -> {w} ({s:.3f})")

# def get_sim_mat(words):
#     sim_mat = {}
#     for a in words:
#         sim_mat[a] = {}
#         for b in words:
#             sim_mat[a][b] = model.similarity(a, b)
#     return sim_mat, words

# def print_sim_mat(sim_mat, words):
        
#     def wpad(w, n):
#         return w[:n] if len(w) >= n else w + "".join([" " for _ in range(n - len(w))])

#     print("\t"+ " ".join([ wpad(b, 6) for b in words]))
#     for a in words:
#         line = []
#         for b in words:
#             s = sim_mat[a][b]
#             d = f"+{s:.3f}" if s >= 0 else f"{s:.3f}"
#             line.append(d)
#         out = " ".join(line)
#         print(f"{a}\t{out}")

# # print_sim_mat(*get_sim_mat([c[0] for c in cards]))

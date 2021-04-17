# Codenames Bot

## Algorithms

### Clue-Giving

Clue-Giver ALGORITHM (don't mind if I do)
Input the board
Input which team to give clues for
For each turn:
    Need to output one word + number of cards as a clue (absolutely -R)
    Receives guesses and marks the word as seen until no of valid guesses is elapsed 
chimken goes to the winner

The Magic Sauce ALGORITHM (perhaps lost, perhaps not)
- Heuristic for the bot to rank its predictions
- Bot needs to be able to evaluate an abitrary set of words up to a certain number against a set of clues

HIGH level approach of botto 
- Gather sets of cards to predict on
- Make several predictions (clue) for each set of cards
- And rank/choose the one most likely to give the highest success rate according to heuristic 
- Bot needs to try and account for negative constraints (avoid opposing colours, neutral, and meanie words)

GREEDY approach: 
- Create the superset of all card combinations 
- Generate N predictions for each set of card combinations indiscriminately 
- Rank predictions and select 

Couple approaches:
- Rank all possible sets of cards, pick the most similar set, then choose a clue for that set
- Among set of cards, choose the top N most similar sets >1 to generate predictions 

Try Word2Vec for generating clues
https://stats.stackexchange.com/questions/267169/how-to-use-pre-trained-word2vec-model

Try Spacy
https://spacy.io/

### Classes of Approaches to Try

- Aggregate clues by hand size, then normalize scores
- Pick hands that are similar, then search for clues
- Find a pure scoring method that drops the least similar subscore in hands of 3+
- Let the user pick how many cards they want a clue for

## Scripts

### Set Up Repository

```bash
# Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
# Download word embeddings
mkdir embeddings
wget -O embeddings/glove.6B.zip http://nlp.stanford.edu/data/glove.6B.zip
unzip embeddings/glove.6B.zip -d embeddings
rm embeddings/glove.6B.zip
# Make model from word embeddings
mkdir models
python3 make_model.py
# Run the main script
python3 main.py
# Deactivate virtual environment
deactivate
```

### Get Cards From Codenames

```js
// Open https://horsepaste.com
// Go to a game
// Click on Spymaster view
// Run this in the browser console:
Array.from(document.querySelectorAll('.word')).map(e => e.ariaLabel).join('\n')
```

### Convert Embeddings to KeyedVector Model

```python
glove_file = "/Users/vineshkannan/Documents/GitHub/codenames/glove.6B/glove.6B.100d.txt"
model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
```

### Make Environment Variables File

```
PORT=5000
ALLOW=http://localhost:5000,http://104.236.21.173:5000
CHROME=/Users/vineshkannan/Documents/dev/chromedriver
```

### Download Chromium Webdriver

```bash
# https://sites.google.com/a/chromium.org/chromedriver/downloads
mkdir driver
wget -O driver/chromedriver.zip https://chromedriver.storage.googleapis.com/90.0.4430.24/chromedriver_linux64.zip
unzip driver/chromedriver.zip -d driver
rm driver/chromedriver.zip
echo -e "CHROME=$(pwd)/driver/chromedriver" >> .env
# If Chrome (in addition to Chrome webdriver) is not installed:
# https://gist.github.com/ziadoz/3e8ab7e944d02fe872c3454d17af31a5
```

### Get Random Words
```js
// https://randomwordgenerator.com/
// Pick nouns, adjectives, or verbs, set word size by syllables or letters
Array.from(document.querySelectorAll(".support")).map(e => e.innerText)
```

## Notes

- Stop words list from [sebleier](https://gist.github.com/sebleier/554280)
- Emoji as favicon from [Lea Verou, Chris Coyier](https://css-tricks.com/emojis-as-favicons)

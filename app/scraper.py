from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from clues import parse_cards_from_board


def get_board_from_horsepaste(url, chrome_path):
    chrome_opts = Options()
    chrome_opts.add_argument("---headless")
    chrome_opts.add_argument("---no-sandbox")
    chrome_opts.add_argument("---disable-dev-shm-usage")
    driver = webdriver.Chrome(chrome_path, options=chrome_opts)
    driver.get(url)
    button = driver.find_element_by_class_name("codemaster")
    button.click()
    els = driver.find_elements_by_class_name("word")
    board = "\n".join([el.get_attribute("aria-label") for el in els])
    cards = parse_cards_from_board(board)
    flipped = "\n".join([c[0] for c in cards if c[2]])
    driver.quit()
    return board, flipped

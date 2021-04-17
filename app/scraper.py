from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from clues import parse_cards_from_board


def get_board_from_horsepaste(url, chrome_path):
    chrome_opts = Options()
    chrome_opts.add_argument("---headless")
    chrome_opts.add_argument("---no-sandbox")
    chrome_opts.add_argument("---disable-dev-shm-usage")
    driver = webdriver.Chrome(chrome_path, options=chrome_opts)
    driver.get(url)
    board = ""
    flipped = ""
    try:
        button = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, "codemaster")))
        # button = driver.find_element_by_class_name("codemaster")
        button.click()
        els = driver.find_elements_by_class_name("word")
        board = "\n".join([el.get_attribute("aria-label") for el in els])
        cards = parse_cards_from_board(board)
        flipped = "\n".join([c[0] for c in cards if c[2]])
    finally:
        driver.quit()
    return board, flipped

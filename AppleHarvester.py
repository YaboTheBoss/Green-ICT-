from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time 

url = "https://www.apple.com/newsroom/search/2019/?q=press+release&page=6"

driver = webdriver.Chrome()  
driver.get(url)

try:
    links = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a.result__item.row-anchor'))
    )

    print(f"Found {len(links)} press release links")
    press_release_urls = [link.get_attribute('href') for link in links]
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    driver.quit()

from collections import defaultdict
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json

COUNTER_FILE = 'press_release_counters.json'

def load_date_counters():
    if os.path.exists(COUNTER_FILE):
        try:
            with open(COUNTER_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_date_counters(counters):
    with open(COUNTER_FILE, 'w') as f:
        json.dump(counters, f)

persistent_counters = load_date_counters()

for url in press_release_urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    date_elem = soup.find('span', class_='category-eyebrow__date')
    if not date_elem:
        print(f"No date found for URL: {url}")
        continue

    pub_date = date_elem.text.strip()
    try:
        date_obj = datetime.strptime(pub_date, "%B %d, %Y")
        formatted_date = date_obj.strftime("%m%d%y")
    except ValueError:
        print(f"Invalid date format for URL: {url}")
        continue
    
    if formatted_date not in persistent_counters:
        persistent_counters[formatted_date] = 0
    persistent_counters[formatted_date] += 1
    
    article_count = persistent_counters[formatted_date]
    save_date_counters(persistent_counters)

    articles = soup.find_all('div', class_='pagebody-copy')
    article_texts = []
    
    for article in articles:
        text = article.text.strip()
        if ".com" in text: 
            continue
        if "Pricing and Availability" in text:
            break
        if "Apple revolutionized" in text:
            break
        if "This press release contains" in text: 
            break
        article_texts.append(text)

    filename = f"{formatted_date}_{article_count}_Apple.txt"

    year = date_obj.strftime("%Y")
    month = date_obj.strftime("%m").lstrip('0')
    company_directory = os.path.join("Apple", year, month)
    os.makedirs(company_directory, exist_ok=True)
        
    article_content = "\n".join(article_texts)

    file_path = os.path.join(company_directory, filename)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(article_content)
        
    print(f"Saved: {filename}")
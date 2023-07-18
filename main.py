from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.firefox import GeckoDriverManager

def get_movie_info(url):
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    title = soup.find('div', {'class': 'TitleBlock__TitleContainer-sc-1nlhx7j-1 jxsVNt'}).h1.text
    storyline = soup.find('span', {'class': 'GenresAndPlot__Plot-cum89p-6 bUyrda'}).text

    print("Title:", title)
    print("Storyline:", storyline)
    print("--------------------------")

# Using Firefox; you can use whichever browser you like
driver = webdriver.Firefox()

# movie_name = input("Enter a movie name: ")
movie_name = "spiderman"
driver.get("https://www.imdb.com/")
search_box = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "suggestion-search")))
search_box.send_keys(movie_name)
search_box.send_keys(Keys.RETURN)

#########################


# Find the search results container
search_results_container = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "findResults")))

# Find the first movie link within the search results container
first_movie_link = search_results_container.find_element(By.CSS_SELECTOR, ".result_text a")
movie_url = first_movie_link.get_attribute("href")

# Click on the first movie link
driver.get(movie_url)

# Get and print the movie information
get_movie_info(driver.current_url)

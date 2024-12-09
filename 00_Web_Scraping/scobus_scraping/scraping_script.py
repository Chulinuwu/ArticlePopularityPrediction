from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import time
import os
from _secret import _id, _pass

def scaping_script():

    driver = webdriver.Chrome()
    
    try:
        driver.get('https://www.scopus.com/search/form.uri?display=basic#basic')
        
        signin_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "signin_link_move"))
        )
        signin_button.click()
        
        accept_cookie = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        )
        accept_cookie.click()
        
        email_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "bdd-email"))
        )
        email_input.send_keys(_id)
        
        continue_button = driver.find_element(By.ID, "bdd-elsPrimaryBtn")
        continue_button.click()
        
        password_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "bdd-password"))
        )
        password_input.send_keys(_pass)
        
        login_button = driver.find_element(By.ID, "bdd-elsPrimaryBtn")
        login_button.click()
        
        time.sleep(5)


        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "styleguide-input-module___SqPU"))
        )
        search_box.send_keys('o')

        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))
        )
        search_button.click()

        time.sleep(5)
        

        WebDriverWait(driver, 50).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        sort_by = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//*[@id='container']/micro-ui/document-search-results-page/div[1]/section[2]/div/div[2]/div/div[2]/div/div[1]/table/tbody/tr/td[3]/div/div/div[1]/label/select"))
        )

        select = Select(sort_by)

        select.select_by_visible_text("Relevance")

        start_year = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//*[@id='year-section']/div/div/div/div[2]/div/div[2]/div/div/div/div/div[1]/div/label/input"))
        )
        start_year.send_keys("2018")
        start_year.send_keys(Keys.TAB)
        end_year = driver.switch_to.active_element
        end_year.send_keys(Keys.BACKSPACE)
        end_year.send_keys(Keys.BACKSPACE)
        end_year.send_keys(Keys.BACKSPACE)
        end_year.send_keys(Keys.BACKSPACE)
        end_year.send_keys(Keys.BACKSPACE)
        end_year.send_keys(Keys.BACKSPACE)

        end_year.send_keys("2023")

        time.sleep(2)

        WebDriverWait(driver, 50).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        export_button = WebDriverWait(driver,10).until(
            EC.presence_of_element_located((By.XPATH, "//button[.//span[text()='Export']]"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", export_button)
        driver.execute_script("arguments[0].click();", export_button)
        csv_button = WebDriverWait(driver,10).until(
            EC.presence_of_element_located((By.XPATH, "//button[.//span[text()='CSV']]"))
        )
        csv_button.click()
        select_all_button = WebDriverWait(driver,10).until(
            EC.presence_of_element_located((By.XPATH, "//button[.//span[text()='Select all information']]"))
        )
        select_all_button.click()
        radio_label = driver.find_element(By.XPATH, "//label[.//span[text()='Documents']]")
        radio_label.click()
        input_from = driver.find_element(By.XPATH, "//input[@placeholder = 'From']")
        input_from.send_keys(1)
        input_to = driver.find_element(By.XPATH, "//input[@placeholder = 'To']")
        input_to.send_keys(2500)
        export_button2 = WebDriverWait(driver,10).until(
            EC.presence_of_element_located((By.XPATH, "//button[.//span[.//div[text()='Export']]]"))
        )
        export_button2.click()

        time.sleep(2) 

        WebDriverWait(driver, 5000).until(
            EC.presence_of_element_located((By.XPATH, "//span[.//div[.//div[text()='Your CSV file was successfully exported.']]]"))
        )
        time.sleep(2)
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        driver.quit()

if __name__ == "__main__":
    if _id and _pass:
        scaping_script()
    else :
        print("Creadential not found! [Following the instruction]")
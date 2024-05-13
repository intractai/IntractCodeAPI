import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from argparse import ArgumentParser

# Parse command-line arguments
parser = ArgumentParser(description="Scrape top PyPI packages")
parser.add_argument('-n', type=int, default=8000, help='Number of packages to output')
args = parser.parse_args()

if args.n > 8000:
    raise ValueError("The maximum number of packages is 8000")

# Setup Chrome options for headless mode
options = ChromeOptions()
options.add_argument("--headless")

# Initialize the WebDriver for Firefox
service = ChromeService(executable_path='chromdriver/chromedriver')  # Update the path to chromedriver
driver = webdriver.Chrome(options=options, service=service)
# Open the webpage
driver.get('https://hugovk.github.io/top-pypi-packages/')

output_file = 'py_libs.txt'
try:
    # Wait for the button to be clickable and click it

    show_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, '//button[@ng-click="show(8000)"]'))
    )
    show_button.click()
    
    # Wait for the items to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, 'two.ng-binding'))
    )
    
    # Scrape the package names
    packages = driver.find_elements(By.CLASS_NAME, 'two.ng-binding')
    package_names = [package.text for package in packages][:args.n]  # Limit the output based on 'n'
    
    # Write the package names to 'requirements.txt'
    with open(output_file, 'w') as file:
        for name in package_names:
            file.write(name + '\n')
    
finally:
    # Close the WebDriver
    driver.quit()

# Notify the user
print(f"Saved the top {args.n} PyPI package names to {output_file}")
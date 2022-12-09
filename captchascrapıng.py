from selenium import webdriver
import chromedriver_autoinstaller
from selenium.webdriver.common.by import By
import time
from PIL import Image

def get_captcha(driver, element, path):
    # now that we have the preliminary stuff out of the way time to get that image :D
    #location = element.location_once_scrolled_into_view
    location = element.location
    size = element.size
    # saves screenshot of entire page
    driver.switch_to.frame(driver.find_element("id","gen__1062"))
    driver.save_screenshot(path)

    # uses PIL library to open image in memory
    image = Image.open(path)

    left = location['x']
    top = location['y']
    right = location['x'] + size['width']
    bottom = location['y'] + size['height']

    image = image.crop((left, top, right, bottom))  # defines crop points
    image.save(path, 'png')  # saves new cropped image
count=0
path="./image"
if not os.path.exists(path):
    if os.name == 'posix':
        !mkdir -p {path}
    if os.name == 'nt':
         !mkdir {path}

lst = os.listdir(path) # your directory path
number_files = len(lst)
countim= input('How many photo you need?')
limit=countim+number_files
while number_files<limit:
    chromedriver = chromedriver_autoinstaller.install()
    driver = webdriver.Chrome(chromedriver)

    try:
        driver.get(url)
        time.sleep(3)
        # download image/captcha
        img = driver.find_element("id","gen__1062")

        print(img.location,img.size)
        get_captcha(driver, img, "Image{}.png".format(number_files))

    except:
        pass

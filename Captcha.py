import cv2
import tensorflow as tf 
tf.config.experimental_run_functions_eagerly(True)
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from selenium import webdriver
import chromedriver_autoinstaller
from selenium.webdriver.common.by import By
import time
from PIL import Image
import os
import numpy as np 
import matplotlib.pyplot as plt
print("Libs imported")
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file("./Model/pipeline.config")
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
print("Pipeline settings...")
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join("./Model/ckpt-7")).expect_partial()
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections



category_index = label_map_util.create_category_index_from_labelmap("./Model/label_map.pbtxt")
print("Model deployed.")
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


chromedriver = chromedriver_autoinstaller.install()
driver = webdriver.Chrome(chromedriver)
url = 'https://ivd.gib.gov.tr/tvd_side/main.jsp?token=d1078f5e3dc646b78d5d4e5842f21e97feb48d366bc7617458b6679dec12675154a01fccc42292bb04d926bc259dbc75e39dd8e202535fd70a7098396c74a6f7'

driver.get(url)
time.sleep(4)
print("Connecting...")
# download image/captcha
img = driver.find_element("id","gen__1062")

print(img.location,img.size)
get_captcha(driver, img, "captcha.png")
#Detecting
print("Detecting...")
    
img = cv2.imread("captcha.png")
image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=6,
            min_score_thresh=.3,
            agnostic_mode=False)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()
dos = detections['detection_boxes'][0:6]
clas=detections['detection_classes']+label_id_offset
clas = clas[0:6]
empty1=0
empty2=0
for i in range(0,20):
    i%=5
    if dos[i,1]>dos[i+1,1]:
        empty1=dos[i,1]
        dos[i,1]=dos[i+1,1]
        dos[i+1,1]=empty1
        empty2=clas[i]
        clas[i]=clas[i+1]
        clas[i+1]=empty2
output=""

for i in clas:
    output="{}{}".format(output,category_index[i]['name'])
driver.switch_to.default_content()
capthaout = driver.find_element("id","gen__1065").send_keys("{}".format(output))

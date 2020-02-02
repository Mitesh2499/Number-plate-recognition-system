import numpy as np
import pandas as pd
import cv2
import time
import imutils
import pytesseract
tessdata_dir_config = '--tessdata-dir "C:\Program Files\Tesseract-OCR\tesseract.exe"'
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# Read the image file
image = cv2.imread('download.jpg')

# Resize the image - change width to 500
image = imutils.resize(image, width=500)

# Display the original image
cv2.imshow("Original Image", image)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("1 - Grayscale Conversion", gray)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
#cv2.imshow("2 - Bilateral Filter", gray)

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
#cv2.imshow("4 - Canny Edges", edged)

# Find contours based on Edges
#new, cnts= cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


(new, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]  # sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None  # we currently have no Number plate contour

# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:  # Select the contour with 4 corners
        NumberPlateCnt = approx  # This is our approx Number Plate Contour
        break


# Drawing the selected contour on the original image
#cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
#cv2.imshow("Final Image With Number Plate Detected", image)

# Masking the part other than the number plate
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
new_image = cv2.bitwise_and(image, image, mask=mask)
cv2.namedWindow("Final_image", cv2.WINDOW_NORMAL)
cv2.imshow("Final_image", new_image)


text = pytesseract.image_to_string(new_image)
print(text)
# Data is stored in CSV file
sys_time = time.asctime(time.localtime(time.time()))
raw_data = {'date': [sys_time],
            'v_number': [text]}

df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
df.to_csv('data.csv')

# rslt_df = df.loc[df[''] == text]  car.jpeg new4.jpg
df = pd.read_excel('society.xlsx', dtype=str)
df.head(5)
res = df.loc[df['v_number'] == text].values
print(res)
name = res[0][1]
print(str(name))

phone_no = res[0][3]
print(str(phone_no))
a = "+91" + str(phone_no)
print(a)
# we import the Twilio client from the dependency we just installed
from twilio.rest import Client

# the following line needs your Twilio Account SID and Auth Token
client = Client("Account SID", "Auth Token")

# change the "from_" number to your Twilio number and the "to" number
# to the phone number you signed up for Twilio with, or upgrade your
# account to send SMS to any phone number
client.messages.create(to="+91" + str(phone_no),
                       from_="Twilio number",
                       body=name + " Your Vehical No " + text + " Just pass society gate at " + sys_time + " !!")
print("sent")
cv2.waitKey(0)  # Wait for user input before closing the images displayed

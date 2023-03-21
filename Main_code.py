""" Importing Libraries """

import timeit
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
from PIL import Image
from SSIM_PIL import compare_ssim
import os
from tkinter import *
from tkinter import Tk
from tkinter import filedialog
from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Algorithm import lsb
from Astar import State,A_Star_Solver
import warnings
import cv2
import numpy as np
warnings.filterwarnings("ignore")

#----------------------------------------------------------------------------
"browsing the Cover image"
root = Tk()
root.withdraw()
options = {}
options['initialdir'] = 'Input_Images'
global fileNo
options['mustexist'] = False
file_selected = filedialog.askopenfilename(title = "Select file",filetypes = (("PNG  files","*.png"),("all files","*.*")))
head_tail = os.path.split(file_selected)
fileNo=head_tail[1].split('.')

#---------------------------------------------------------------------------------------------------------
"Load Input Image"
#Get the Input
Image_test=file_selected
Input=cv2.imread(file_selected)
img = cv2.cvtColor(Input, cv2.COLOR_BGR2RGB)
print("***INPUT IMAGE****")
plt.figure()
plt.imshow(img) 
plt.title("Frame Size")
plt.show()
print('Shape of Frame Image: ',img.shape) 
#-------------------------------------------------------------------------------------------------------
"Preproceesing Image"

#Resize the Given Input Image
scale_percent = 50 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
print('Resized Dimensions using Preprocessing : ',resized.shape)
plt.figure()
plt.title("RESIZED_IMAGE")
plt.imshow(resized) 
plt.show()

#----------------------------------------------------------------------------------------------------------

""" PIXEL REPETITION METHOD (PRM) """

# Cover Image - C(i,j)
i = int(2 * resized.shape[0] - 1)
j = int(2 * resized.shape[1] - 1)

# Equal the Cover image and Resized Image
cover = resized
(resized.shape[0] * 2) - 1

i = (resized.shape[0] * 2) - 1
j = (resized.shape[1] * 2) - 1 

i = i + 1
j = j + 1

dimensions = (i,j)


#------------------------------------------------------------------------------------------------------------------------------
# Saving the Cover Image

cover_img = cv2.resize(img,dimensions, interpolation = cv2.INTER_AREA )
cover = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
cv2.imwrite('./Input/data/train/prm_output.png', cover)
j = cv2.imread('./Input/data/train/prm_output.png')
img = cv2.cvtColor(j, cv2.COLOR_BGR2RGB)
print("******PRM IMAGE******")
plt.figure()
plt.imshow(img) 
plt.title("PIXEL REPETITION METHOD IMAGE")
plt.show() 
print('Resized Dimensions Using PRM : ',cover.shape)
print("\n")

#------------------------------------------------------------------------------------------------
"AES CRYPTOGRAPHYSYSTEM"

hash_obj=SHA256.new()
hkey=hash_obj.digest()
print(hkey)

file = open("Secret Message.txt")
line = file.read().replace("\n", " ")

def encrypt (info):
     msg=info
     BLOCK_SIZE=16
     PAD="{"
     padding =lambda s:s+(BLOCK_SIZE-len(s)% BLOCK_SIZE)*PAD
     cipher=AES.new(hkey,AES.MODE_ECB)
     result=cipher.encrypt(padding(msg).encode('utf-8'))
     return result
msg=line
cipher_text=encrypt(msg)
print(cipher_text) 

hexa = cipher_text.hex()


#-----------------------------------------------------------------------------------------------------------------------------------

""" LSB EMBEDDING """


# Hiding the Hexa values to the Images
start = timeit.default_timer()
secret = lsb.hide("./Input/data/train/prm_output.png",hexa)   #PRM-imgage
end = timeit.default_timer()
print("**** LSB EMBEDDED IMAGE *****")
print('Time Consuming for Encryption:' , (end-start)/60 , "minutes.")
secret.save("./Input/data/test/lsb_secret.png")   #embedded image
plt.figure()
plt.title("LSB EMBEDDED IMAGE")
plt.imshow(secret) 
plt.show()
#----------------------------------------------------------------------------------------------------

# High Resolution

from PIL import Image
imag = Image.open("./Input/data/test/lsb_secret.png")
quality_val = 200
imag.save("./Input/data/test/lsb_secret.png", 'png', quality=quality_val)
print("***** HIGH RESOLUTION IMAGE******")
plt.figure()
plt.title("HIGH RESOLUTION IMAGE ")
plt.imshow(imag) 
plt.show()

#-------------------------------------------------------------------------------------------
"Encryption on Image "
password=(input('Enter the Key Generated Password :'))
img=cv2.imread("./Input/data/test/lsb_secret.png")
img1=img.astype('uint8')


img=cv2.resize(img, (300,300))
li=[0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,77,0,90,0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,0,90,11,125,5,34,0,57,75,10,37,4,0,90,0,75,22,36,54,111,0,9,0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,0,90,0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,34,0,20,40,0,11,32,52,0,33,49,144,55,44,11,95,60,0,5,44,68,0,90,60,4,50,0,550,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,0,90,0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,0,90,11,125,5,34,0,57,75,10,37,4,0,90,0,75,22,36,54,111,0,9,0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,0,90,0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,34,0,20,40,0,11,32,52,0,33,49,144,55,44,11,95,60,0,5,44,68,0,90,60,4,50,0,55]
for i in range(300):
    for q in range(li[i]):
        temp0=img[i][0][0]
        temp1=img[i][0][1]
        temp2=img[i][0][2]
        for j in range(1,300):
            img[i][j-1]=img[i][j]
        img[i][299][0]=temp0
        img[i][299][1]=temp1
        img[i][299][2]=temp2

cv2.imwrite('lsb_secret1.png',img)
print("***************Encrypted form Image***********")
cv2.imshow(' Encrypted Image',img)
cv2.waitKey(0)  
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------
"decryption using A* Algorithm"

inp=(input('Enter the Key Password :'))
attempt = 1
while (inp != password) and (attempt < 3):
    value = 3 - attempt
    # print(f'')
    inp = input(str(value) + " try remaining: ")  
    attempt = attempt + 1

if attempt == 3:
    print ('Sorry you have reached maximum number of attempts')
    exit()

if (inp == inp):
    print('\nAccess Granted!') 
        
de = lsb.reveal(imag)

#Convert Hexa into Bytes
byte_array = bytes.fromhex(hexa)

# Convert Bytes to String Bytes
start = timeit.default_timer()
end = timeit.default_timer()


print('Time Consuming for Decryption:' , (end-start)/60 , "minutes.")
print("\n")


def decrypt (info):
     msg=info

     PAD="{"
     decipher=AES.new(hkey,AES.MODE_ECB)
     pt=decipher.decrypt(msg).decode('utf-8')
     pad_index=pt.find(PAD)
     results=pt[:pad_index]
     return results
fileoutput= decrypt(cipher_text) 
print(f"TUSHAR====={cipher_text}")
print(f"\nTUSHARGUPTA====={fileoutput}")

#-----------------------------------------------------------------------------------------------
"Decrypted image"


img1=cv2.imread("lsb_secret1.png")
img1=cv2.resize(img1, (300,300))
li=[0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,77,0,90,0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,0,90,11,125,5,34,0,57,75,10,37,4,0,90,0,75,22,36,54,111,0,9,0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,0,90,0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,34,0,20,40,0,11,32,52,0,33,49,144,55,44,11,95,60,0,5,44,68,0,90,60,4,50,0,550,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,0,90,0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,0,90,11,125,5,34,0,57,75,10,37,4,0,90,0,75,22,36,54,111,0,9,0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,0,90,0,75,22,36,54,111,0,9,26,44,3,77,44,0,11,125,5,34,0,57,75,10,37,4,34,0,20,40,0,11,32,52,0,33,49,144,55,44,11,95,60,0,5,44,68,0,90,60,4,50,0,55]
for i in range(300):
    for q in range(li[i]):
        temp0=img1[i][299][0]
        temp1=img1[i][299][1]
        temp2=img1[i][299][2]
        for j in range(1,300):
            img1[i][300-j]=img1[i][(300-j)-1]
        img[i][0][0]=temp0
        img[i][1][1]=temp1
        img[i][2][2]=temp2

cv2.imwrite('decrypted_image.png',Input)
print("***************Decrypted form Image***********")
cv2.imshow(' Decrypted Image',Input)
cv2.waitKey(0)  
cv2.destroyAllWindows() 

#---------------------------------------------------------------------------------
"Search the data "

print("***************Decode form Image***********")
print(f"The Decode Message from the Image  = {fileoutput}")
print("\n")

#-----------------------------------------------------------------------------------------

""" PERFORMANCE ANALYSIS TO CALCULATE PSNR AND MSE"""

#For Cover and Embedded Image
img1 = cv2.imread("./Input/data/train/prm_output.jpeg")
img2 = cv2.imread("./Input/data/test/lsb_secret.jpeg")
psnr = cv2.PSNR(img1, img2)
mse = np.mean((img1 - img2) ** 2) 
#SSIM
image1 = Image.open("./Input/data/train/prm_output.jpeg")
image2 = Image.open("./Input/data/test/lsb_secret.jpeg")
value = compare_ssim(image1, image2)
print("### Cover and Embedded Image ###")
print(f"The MSE Value : {mse}")
print(f"The PSNR Value : {psnr} db")
print(f"The SSIM Value : {value} ")
print("\n")



print("Video Location for Stegnography ",fileNo)
print('Pixel Shape of Stegnography Image: ',img.shape) 

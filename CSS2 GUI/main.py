# Import Required Libraries

from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

from customtkinter import *
import cv2
import tensorflow as tf
import numpy as np
from PIL import ImageTk, Image
import os

#If Image Loaded from previous run remove it
if os.path.exists("loadedIMG.jpg"):
  os.remove("loadedIMG.jpg")

#Load Model
model = tf.keras.models.load_model('denseNet3.h5')

#Skin Lesion Classes
classes = [
    'Actinic Keratoses and Intrarpithelial Carcinoma', 
    'Basal Cell Carcinoma', 
    'Benign Keratosis Lesion', 
    'Dermatofibroma', 
    'Melanoma', 
    'Malanocytic Nevi', 
    'Vascular Lesions'
]

#Descriptions for the classes
descriptions = [
    'Actinic Keratoses is a type of precancer. Without treatment, AK\n can lead to a type of skin cancer called Intrarpithelial \nsquamous cell carcinoma.',
    'Basal cell carcinoma is the most common type of skin cancer.\nEarly detection and treatment are important to prevent it from spreading or causing damage.',
    'Benign Keratosis Lesions are non-cancerous growths on the skin.\nWhile they can resemble skin cancer, they are harmless and \ntypically dont require treatment unless they become irritated or bothersome.',
    'Dermatofibroma is a benign skin growth that is usually harmless \nand dont require treatment, they can sometimes be confused with \nother more serious skin conditions, so its important to get any \nnew or changing skin growths checked by a dermatologist.',
    'Melanoma can be a life-threatening cancer if not treated early, \nso its crucial to monitor your skin for any changes and seek prompt \nmedical attention if you notice any suspicious growths.',
    'Melanocytic nevi, commonly known as moles, are benign growths on \nthe skin. While most moles are harmless, some may develop into melanoma.\nHave them evaluated by a dermatologist if there are any concerns.',
    'Vascular lesions are abnormalities of blood vessels in the skin that \ncan manifest in various forms. Treatment options depend on the type and \nseverity of the vascular lesion and may include laser therapy, surgery, or medication.'
]


# Create a Window.
MyWindow = CTk() # Create a window
MyWindow.title("Skin Lesion Classifier") # Change the Title of the GUI
MyWindow.geometry('700x700') # Set the size of the Windows

#Create Frame for image to be displayed in
imageframe = CTkFrame(master=MyWindow, fg_color='#8D6F3A', border_color='#FFCC70', border_width=2, width=600, height=450)
imageframe.grid(row=0,column=0, padx=50, pady=25)

#Create Frame for buttons to be displayed in
buttonframe = CTkFrame(master=MyWindow, border_width=2, width=200, height=50)
buttonframe.grid(column=0,row=1)

#Create Frame for class + desciption to be displayed in
classificationframe = CTkFrame(master=MyWindow, fg_color='#C0C0C0', border_color='#FFCC70', border_width=2, width=400, height=100)
classificationframe.grid(column=0,row=2, pady=10)

#Initialise classification label
ClassficationResultLabel = CTkLabel(master=classificationframe, text = "", font=("Arial Bold", 15), text_color='#222222')
ClassficationResultLabel.pack(expand=True)

#Initialise image
image = ImageTk.PhotoImage(Image.open("NOIMAGE.jpg"))
imagebox = CTkLabel(master=imageframe, image = image, text='')
imagebox.pack(expand=TRUE, padx=5, pady=5)


# Open Image Function using OpenCV
def openImg(filename):
    img = cv2.imread(filename) #Read Image
    image = ImageTk.PhotoImage(Image.open(filename)) #Open image using ImageTk
    imagebox.configure(image=image) #Set image in frame to opened image
    imagebox.photo = image
    img = cv2.resize(img, (224,224)) #Resize to 224,224,3 for the model
    cv2.imwrite('loadedIMG.jpg', img) #Save image


# Create Event Methods attached to the button
def BttnOpen_Clicked():
    # Use the File Dialog component to Open the Dialog box to select files
    file = filedialog.askopenfilename(filetypes = (("Images files","*.jpg"),("Video Files","*.mp4"),("all files","*.*")))
    messagebox.showinfo("File Selected", file)
    openImg(file) # Passing the file to openImg method to show is using opencv (imread, imshow)
    

def BttnProcess_Clicked():
    img = cv2.imread('loadedIMG.jpg') #Load image
    img = np.array(img) #Convert to numpy array
    result = model.predict(img[None,:,:]) #Resize image to None, 224,224,3
    finalResult = classes[int(np.argmax(result))] #Result output as array, take highest probability to final result
    desc = descriptions[int(np.argmax(result))] #Get corresponding description
    resultText = f"Classification Result: {finalResult}. \n\n{desc}"
    ClassficationResultLabel.configure(text = resultText)  # Update the Label text on the Window
    
    

#Load Buttons
openBttn = CTkButton(master= buttonframe, text="Open Image", command=BttnOpen_Clicked)
openBttn.pack(expand=True, side=LEFT, padx=15, pady=2)

openProcess = CTkButton(master= buttonframe, text="Process Image", command=BttnProcess_Clicked)
openProcess.pack(expand=True, side=RIGHT, padx=15, pady=2)


# Calling the maninloop()
MyWindow.mainloop()

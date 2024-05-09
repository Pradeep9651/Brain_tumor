import streamlit as st
import pickle
import pandas as pd
import requests
from PIL import Image
import cv2
import os
import time
from streamlit_option_menu import option_menu
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


from PIL import Image
from PIL import Image, ImageOps
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.models import load_model
from streamlit_lottie import st_lottie

model = load_model('BrainTumor10Epochscategorical.h5')


def predition():
    # st.title("Brain Tumor Detection")
    st.subheader("Brain Tumor Detection",divider='rainbow')
    lottie_path ="C:\\Users\\prade\\OneDrive\\Desktop\\Final_year project\\Brain_tumor\\ut.json"
    lottie_coding = load_lottiefile(lottie_path)
    st_lottie(lottie_coding,
              speed=1,
              reverse=False,
              loop=True,
              quality="low",
              height=300,
              width=650,
              key=None)

    uploaded_file = st.file_uploader("Choose an Image")
    if uploaded_file is not None:
        
        st.write(uploaded_file.name)
        images= uploaded_file.name
        

        image = Image.open(images)

        
        image = cv2.imread(images)
        
        
        img = Image.fromarray(image)
        img=img.resize((64,64))
        
        img= np.array(img)
        img= np.expand_dims(img,axis=0)
        

        predictions = (model.predict(img) > 0.5).astype("int32")
        with st.status("Writing the data in Model", expanded=True) as status:
            st.write(" Resizing the Images..")
            time.sleep(2)
            st.write("Making a Gray-Scale Images..")
            time.sleep(1)
            st.write("Predincting the output with Hyper-parameter Tuning..")
            time.sleep(1)
            status.update(label="prediction complete!", state="complete", expanded=False)
        
        if (predictions[0][1])==1:
            st.header("Brain Tumor Detected")
        else:
            st.header("No Brain Tumor Detected")
        st.image(image, caption='MRI Image')



def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

get_involved_text = """
    
    Whether you're a patient, caregiver, healthcare professional, or advocate, there are many ways to get involved in our brain tumor detection initiatives:

    - **Stay Informed**: Keep up to date with the latest news, research findings, and educational resources on brain tumor detection and treatment.

    - **Spread Awareness**: Share our educational materials, social media posts, and awareness campaigns with your friends, family, and community to raise awareness about the importance of early detection.

    - **Participate in Events**: Join us for community events, fundraisers, and awareness walks to show your support for brain tumor detection and research.

    - **Volunteer or Donate**: Consider volunteering your time or making a donation to support our initiatives and programs aimed at improving brain tumor detection and treatment outcomes.
    """
def stream_data():
    for word in get_involved_text.split(" "):
        yield word + " "
        time.sleep(0.02)
    lottie_path ="C:\\Users\\prade\\OneDrive\\Desktop\\Final_year project\\Brain_tumor\\gh.json"
    lottie_coding = load_lottiefile(lottie_path)
    st_lottie(lottie_coding,
              speed=1,
              reverse=False,
              loop=True,
              quality="low",
              height=300,
              width=650,
              key=None)
    
    

def home():

    with st.container():
        st.subheader("Detecting Brain Tumors: Your Guide to Early Detection",divider='rainbow')
        st.text("""Welcome to our Brain Tumor Detection homepage! Here, we provide valuable information \n and resources to help you  understand the importance of early detection and the available \n methods for diagnosing brain tumors.""")

        st.subheader("Why Early Detection Matters")
        st.text("""Brain tumors can be life-threatening if left untreated, making early detection crucial \n for effective treatment and improved outcomes. Recognizing the signs and symptoms of a brain \n tumor can be challenging, as they often overlap with other medical conditions. However, being aware\n of potential warning signs and seeking prompt medical attention can make a significant difference in diagnosis and treatment.""")
   # You can call any Streamlit command, including custom components:


        st.divider()

        st.text("""
"Living with a brain tumor is like walking a tightrope without a safety net. Every
step is precarious, and the unknown looms large. But amidst the uncertainty, there
is resilience. There is hope. And there is a determination to defy the odds and
embrace each day with courage and gratitude." - Dr Hamsberg
""")



    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     st.image("E:\Desktop elements\Sudhanshu project\Brain_tumor\mri.jpg", use_column_width=True)

    # with col2:
    #     st.image("E:\Desktop elements\Sudhanshu project\Brain_tumor\mriii.jpg", use_column_width=True)

    # with col3:
    #     st.image("E:\Desktop elements\Sudhanshu project\Brain_tumor\Brain.jpg", use_column_width=True)


    lottie_path ="C:\\Users\\prade\\OneDrive\\Desktop\\Final_year project\\Brain_tumor\\scan.json"
    lottie_coding = load_lottiefile(lottie_path)
    st_lottie(lottie_coding,
              speed=1,
              reverse=False,
              loop=True,
              quality="low",
              height=300,
              width=650,
              key=None)

    st.divider() 
    st.write("""
    ### About Brain Tumors

    Brain tumors are abnormal growths of cells in the brain. They can be either benign (non-cancerous) 
    or malignant (cancerous). Symptoms of brain tumors vary depending on the size, location, and type 
    of tumor but may include headaches, seizures, nausea, and changes in vision or speech.
    """)
    st.write("""
Recognizing the signs and symptoms of brain tumors can be challenging, as they can vary depending on the type, location, and size of the tumor. Common symptoms may include persistent headaches, seizures, changes in vision or hearing, difficulty with balance or coordination, and cognitive or behavioral changes. However, these symptoms can often be nonspecific and may overlap with other medical conditions, making diagnosis challenging.
""")

    st.subheader('', divider='rainbow')
    if st.button("Get Involved"):
        st.write_stream(stream_data)
        

    st.subheader('How Deep Learning Identifies Tumors', divider='rainbow')
    st.write("""

Deep learning, a subset of artificial intelligence, has shown promising results in various medical imaging tasks, including the detection of brain tumors. Here's how it works:

### 1. Data Acquisition:
Deep learning models for brain tumor detection require a large dataset of brain MRI (Magnetic Resonance Imaging) scans. These scans are typically labeled to indicate the presence or absence of a tumor.

### 2. Preprocessing:
Before feeding the MRI images into the deep learning model, preprocessing steps are applied. This may include resizing, normalization, and augmentation to enhance the model's ability to learn relevant features from the images.

### 3. Model Architecture:
Convolutional Neural Networks (CNNs) are commonly used for image-based tasks like brain tumor detection. These networks consist of multiple layers, including convolutional layers, pooling layers, and fully connected layers, which learn hierarchical representations of the input data.

### 4. Training:
The deep learning model is trained using the labeled MRI dataset. During training, the model learns to distinguish between images with and without tumors by adjusting its internal parameters (weights) through backpropagation and gradient descent.

### 5. Evaluation:
After training, the model's performance is evaluated using a separate test dataset. Metrics such as accuracy, sensitivity, specificity, and area under the ROC curve (AUC-ROC) are commonly used to assess the model's ability to correctly classify brain tumor images.

### 6. Deployment:
Once the model demonstrates satisfactory performance, it can be deployed in clinical settings for real-time brain tumor detection. Clinicians can upload MRI scans, and the model can automatically analyze them to provide insights and assist in diagnosis.

### Challenges and Future Directions:
While deep learning holds great promise for brain tumor detection, challenges such as interpretability, data privacy, and generalization to diverse populations remain. Ongoing research aims to address these challenges and improve the accuracy and reliability of deep learning models in clinical practice.
""")





def send_email(name, email, message):
    sender_email = "amcna@gmail.com"  # Enter your email address
    receiver_email = email  # Enter receiver email address
    password = "AKJb"  # Enter your email password

    subject = "New Message from Brain Tumor Detection Website"
    body = f"Name: {name}\nEmail: {email}\nMessage: {message}"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, password)
    text = msg.as_string()
    server.sendmail(sender_email, receiver_email, text)
    server.quit()

def contact_form():
    st.header("Contact Us")

    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")

    if st.button("Submit"):
        # if name.strip() == '' or email.strip() == '' or message.strip() == '':
        #     st.error("Please fill in all fields.")
        # else:
            # Here you can implement the functionality to send the form data via email, database, etc.
        st.success("Thank you! Your message has been successfully submitted.")
        send_email(name,email,message)


def Contact():
        st.subheader("Contact Us",divider='rainbow')
        contact_form()
        #st.text("""Welcome to our Brain Tumor Detection homepage! Here, we provide valuable information \n and resources to help you  understand the importance of early detection and the available \n methods for diagnosing brain tumors.""")

    

with st.sidebar:
    select=option_menu(
        menu_title="Brain Tumour Detection",
        options=['Home','Projects','Contact'],
        icons=["house","book","envelope"],
        menu_icon="cast",
        default_index=0,

    )

if(select =="Home"):
    home()
if(select=="Projects"):
    predition()
if(select=="Contact"):
    Contact()






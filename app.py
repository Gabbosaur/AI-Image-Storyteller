import streamlit as st
import cv2
import numpy as np
import datetime
# serve per scaricà il modello da hf e utilizzarlo
from transformers import pipeline
from PIL import Image
from gpt4all import GPT4All


def main():
    st.title("✨ Image Storyteller by Gab")
    option = st.radio("Choose an option:", ["Webcam Capture", "Image Upload"])
    if option == "Webcam Capture":
        selected_image = run_webcam()
    elif option == "Image Upload":
        selected_image = run_image_upload()

    # Generate story button
    if st.button("Generate Story"):
        generate_story(selected_image)

def run_webcam():
    picture = st.camera_input("Take a picture")
    if picture:
        img = Image.open(picture)
        # To convert PIL Imag to numpy array:
        img_array = np.array(img)
        pil_img = Image.fromarray(img_array)
        # Save picture to disk
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"captured_image_{timestamp}.jpg"
        save_path = "pictures/" + filename
        cv2.imwrite(save_path, img_array)
        # save_as_jpeg(img_array, save_path)
        return pil_img


def run_image_upload():
    uploaded_file = st.file_uploader(
        "Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        return img


def save_as_jpeg(frame, filename):
    # Save the frame as a JPEG image
    cv2.imwrite(filename, frame)


def img2text(filename):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(filename)[0]["generated_text"]

    # print(text)
    return text


def generate_story(image):
    if image:
        scenario = img2text(image)
        st.write("So this is what I am seeing.. " + scenario)
        st.write("Alright, I will generate a story for you based on that..")
        # system_template = "Sai raccontare le storie; traduci in italiano l'input e genera una breve storia in italiano basandoti sull'input dato, incentrando maggiormente la storia sull'input. La storia generata non deve superare le 30 parole."
        system_template = "You are a story teller; You can generate a short story based on a fascinating narrative, mostly focused on the input. The story should not be more than 30 words."
        # prompt_template = "UTENTE: Ecco l'input da cui partire la storia: {0}\nLA STORIA: "
        prompt_template = "USER: Here's an input to start the story with: {0}\nASSISTANT: "
        llm = GPT4All(
            "C:/Users/eguogab/AppData/Local/nomic.ai/GPT4All/mistral-7b-instruct-v0.1.Q4_0.gguf")
        with llm.chat_session(system_template, prompt_template):
            story = llm.generate(scenario)
            # print(story)
            st.success(story)
    else:
        st.warning("Please add an image first")


if __name__ == "__main__":
    main()

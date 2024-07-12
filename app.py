import streamlit as st
import cv2
import numpy as np
import datetime
import os
import requests
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from PIL import Image
from gpt4all import GPT4All
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

load_dotenv(find_dotenv())
EL_LABS_API_KEY = os.getenv('EL_LABS_API_KEY')
# preload_models()


def main():
    st.title("✨ Image Storyteller by Gab")

    # Select the input mode
    option = st.sidebar.selectbox("Select Input Mode", ["Webcam Capture", "Image Upload"], index=1)
    if option == "Webcam Capture":
        selected_image = run_webcam()
    elif option == "Image Upload":
        selected_image = run_image_upload()

    # Select the language
    language = st.sidebar.selectbox(
        "Select Language", ("English", "Italian"), index=0, disabled=True
    )

    # Select the audio generation model
    audioModel = st.sidebar.selectbox("Select text2speech model", ["LocalBark", "ELabsAPI", "None"], index=2)

    # Generate story
    if st.button("Generate Story"):
        selected_language = language
        generated_story = generate_story(selected_image, selected_language)
        print(generated_story)


        if audioModel == "LocalBark":
            with st.spinner('Generating audio... (this might take a while. Check progress on terminal)'):
                audio_array = generate_audio(generated_story)
                write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
                st.audio("bark_generation.wav")
        elif audioModel == "ELabsAPI":
            with st.spinner('Generating audio...'):
                generate_audio_with_api(generated_story, selected_language)

        elif audioModel == "None":
            pass


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
        return pil_img


def run_image_upload():
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        return img


def img2text(filename):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(filename)[0]["generated_text"]
    return text


def generate_story(image, language):
    if image:
        with st.spinner('Generating...'):
            scenario = img2text(image)
            st.write("So this is what I am seeing.. " + scenario)
            st.write("Alright, I will generate a story in " + language + " for you based on that..")
            if language == "English":
                system_template = "You are a story teller; You can generate a short story based on a fascinating and uplifting narrative with an interesting plot twist; Mostly focused on the input; Do not put any name in it. The story should not be more than 50 words."
                prompt_template = "USER: Here's the input to start the story with: {0}\nASSISTANT:"
            elif language == "Italian":
                system_template = "Sai raccontare le storie; traduci in italiano l'input e genera una breve storia in italiano basandoti sull'input dato, incentrando maggiormente la storia sull'input. Non menzionare nomi propri. La storia generata non deve superare le 30 parole."
                prompt_template = "UTENTE: Ecco l'input da cui partire la storia: {0}\nASSISTENTE: "
            llm = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")
            with llm.chat_session(system_template, prompt_template):
                story = llm.generate(scenario)
                st.success(story)
                return story
    else:
        st.warning("Please add an image first")


def generate_audio_with_api(message, language):
    if language == "English":
        voice_id = "zcAOhNBS3c14rBihAFp1" # "21m00Tcm4TlvDq8ikWAM"
    elif language == "Italian":
        voice_id = "pNInz6obpgDQGcFmaJgB"
    url = "https://api.elevenlabs.io/v1/text-to-speech/" + voice_id

    payload = {
        "model_id": "eleven_monolingual_v1",
        "text": message,
        "voice_settings": {
            "similarity_boost": 0,
            "stability": 0,
            "style": 0,
            "use_speaker_boost": True
        }
    }
    headers = {"Content-Type": "application/json",
               'xi-api-key': EL_LABS_API_KEY,
               'accept': 'audio/mpeg'}

    response = requests.request("POST", url, json=payload, headers=headers)

    if response.status_code == 200 and response.content:
        with open('gen-audio.mp3', 'wb') as f:
            f.write(response.content)
        st.audio('gen-audio.mp3')
        return response.content


if __name__ == "__main__":
    main()

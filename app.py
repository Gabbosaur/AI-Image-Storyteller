import streamlit as st
import cv2
import numpy as np
import datetime
import os
import requests
from groq import Groq
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from PIL import Image
from gpt4all import GPT4All
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

load_dotenv(find_dotenv())
EL_LABS_API_KEY = os.getenv('EL_LABS_API_KEY')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
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
        "Select Language", ("English", "Italian"), index=0, disabled=False
    )

    # Select the audio generation model
    audioModel = st.sidebar.selectbox("Select text2speech model", ["LocalBark", "ELabsAPI", "None"], index=2)

    # Select the LLM platform
    llmPlatform = st.sidebar.selectbox("Select LLM platform", ["GPT4ALL", "Ollama", "Groq"], index=2)

    # Generate story
    if st.button("Generate Story"):
        selected_language = language
        if llmPlatform == "GPT4ALL":
            print("✨ Generating story with GPT4ALL...")
            generated_story = generate_story_with_gpt4all(selected_image, selected_language)
        elif llmPlatform == "Ollama":
            print("✨ Generating story through Ollama...")
            generated_story = generate_story_with_chatOllama(selected_image, selected_language)
        elif llmPlatform == "Groq":
            print("✨ Generating story through Groq...")
            generated_story = generate_story_with_groq(selected_image, selected_language)
        print(generated_story)
        if generated_story:
            st.success(generated_story)


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
        "image-to-text", model="Salesforce/blip-image-captioning-large")
        # "image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(filename)[0]["generated_text"]
    return text

def remove_arafed(text):
    cleaned_text = text.replace("arafed", "").replace("araffed", "")
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

def generate_story_with_chatOllama(image, language):
    if image:
        with st.spinner('Generating...'):
            scenario = img2text(image)
            scenario = remove_arafed(scenario)
            st.write("So this is what I am seeing... ")
            st.info(scenario)
            st.write("Alright, I will generate a story in " + language + " for you based on that..")

            try:
                # Instantiate the ChatOllama object
                llm = ChatOllama(base_url=OLLAMA_BASE_URL, model="mistral:7b", temperature=1)
                # llm = ChatOllama(base_url="http://10.8.0.106:50000", model="llama3.2:3b", temperature=1)
                if language == "English":
                    # Create the prompt template with the required data
                    messages = [
                        SystemMessage(content="As a master storyteller and motivational speaker, your task is to craft captivating and inspiring stories based on a given simple phrase. The stories should be concise, not exceeding 100 words, and should not contain any names. Your stories should have the power to ignite people's imagination and motivate them to constantly improve themselves. Remember, the key is to create a brief yet impactful narrative that leaves a lasting impression. Begin now."),
                        HumanMessage(content="Here's the phrase: " + scenario)
                    ]
                elif language == "Italian":
                    # Create the prompt template with the required data
                    messages = [
                        SystemMessage(content="As an Italian-speaking AI, your role is to act as a history Italian teacher and motivational speaker. Your task is to create innovative and inspiring stories based on a simple phrase in Italian. The stories must be translated to Italian and should be concise, not exceeding 100 words, and should not contain proper names. They should stimulate people's imagination and consistently motivate them to improve themselves. Remember, the key is to create a short yet impactful narrative that leaves a lasting impression and must be in Italian language. Please don't make a numbered list. Don't put any English texts."),
                        HumanMessage(content="Ecco la frase: " + scenario)
                    ]

                # Get the response
                story = llm.invoke(messages)
                story = story.content
            except Exception as e:
                print(f"An error occurred while generating the story: {e}")
                st.error("An error occurred while generating the story.")
                return ""

    return story

def generate_story_with_gpt4all(image, language):
    if image:
        with st.spinner('Generating...'):
            scenario = img2text(image)
            scenario = remove_arafed(scenario)
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
                return story
    else:
        st.warning("Please add an image first")

def generate_story_with_groq(image, language):
    if image:
        with st.spinner('Generating...'):
            scenario = img2text(image)
            scenario = remove_arafed(scenario)
            st.write("So this is what I am seeing.. " + scenario)
            st.write("Alright, I will generate a story in " + language + " for you based on that..")

            if language == "English":
                system_prompt = "As a master storyteller and motivational speaker, your task is to craft captivating and inspiring stories based on a given simple phrase. The stories should be concise, not exceeding 100 words, and should not contain any names. Your stories should have the power to ignite people's imagination and motivate them to constantly improve themselves. Remember, the key is to create a brief yet impactful narrative that leaves a lasting impression. Begin now."
            elif language == "Italian":
                system_prompt = "In qualità di maestro narratore e oratore motivazionale, il tuo compito è creare storie coinvolgenti e ispiratrici basate su una semplice frase fornita. Le storie devono essere concise, non superare le 100 parole e non contenere nomi. Devono avere il potere di accendere l'immaginazione delle persone e motivarle a migliorarsi continuamente. Ricorda, la chiave è creare un racconto breve ma incisivo, capace di lasciare un'impressione duratura. Comincia ora."
            client = Groq(
                api_key=GROQ_API_KEY,
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": "Here's the scenario: " + scenario,
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content


def generate_audio_with_api(message, language):
    if language == "English":
        voice_id = "NOpBlnGInO9m6vDvFkFC"
    elif language == "Italian":
        voice_id = "uScy1bXtKz8vPzfdFsFw"

    client = ElevenLabs(
        api_key=EL_LABS_API_KEY,
    )
    response = client.text_to_speech.convert(
        voice_id=voice_id,
        # voice_id="pNInz6obpgDQGcFmaJgB", # Adam pre-made voice
        output_format="mp3_22050_32",
        text=message,
        model_id="eleven_turbo_v2_5", # use the turbo model for low latency
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )
    # uncomment the line below to play the audio back
    # play(response)
    # Generating a unique file name for the output MP3 file
    save_file_path = "gen-audio.mp3"
    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    st.audio("gen-audio.mp3")


if __name__ == "__main__":
    main()

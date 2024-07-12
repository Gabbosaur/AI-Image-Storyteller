# ✨ Image Storyteller with AI
You take a picture and the AI will tell you story about it :)

# Installation
Prerequisites
- Python version ≥ 3.10

Clone the repo
```bash
git clone https://github.com/Gabbosaur/AI-Image-Storyteller.git
```
Navigate to the cloned repo
```bash
cd AI-Image-Storyteller
```
Create a Python virtual environment
```bash
python -m venv myenv
```
Activate your virtual environment
```bash
source myenv/Scripts/activate
```
Install the dependencies
```bash
pip install -r requirements.txt
```

# How to Use
Run the app. Take note that it will take some time to download the models on the first run.
```bash
streamlit run app.py
```

The main page will appear and you can take a picture from your webcam or upload an existing one.<br>
Then click on _Generate Story_ and wait for the output ✨

If you plan to use ElevenLabs API  for text2speech, make sure to create a .env file in the main folder with its API key inside.

```code
EL_LABS_API_KEY = "<your_elevenlabs_api_key>"
```
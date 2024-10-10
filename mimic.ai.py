# Import necessary libraries
import torch
import numpy as np
from pathlib import Path
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
import librosa
import soundfile as sf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import base64
from pydub import AudioSegment

app = Flask(__name__)
# Define the paths to the pre-trained models
encoder_model_fpath = Path("weights//encoder.pt")
synthesizer_model_fpath = Path("weights//synthesizer.pt")
vocoder_model_fpath = Path("weights//vocoder.pt")

# Load the models
print("Loading the encoder model...")
encoder.load_model(encoder_model_fpath)

print("Loading the synthesizer model...")
synthesizer = Synthesizer(synthesizer_model_fpath)

print("Loading the vocoder model...")
vocoder.load_model(vocoder_model_fpath)

# Function to generate audio from text
def generate_audio(text, reference_audio_path):
    # Preprocess the reference audio file
    preprocessed_wav = encoder.preprocess_wav(reference_audio_path)
    
    # Generate the speaker embedding
    embed = encoder.embed_utterance(preprocessed_wav)
    
    # Generate the mel spectrogram
    texts = [text]
    embeds = [embed]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    
    # Generate the waveform
    generated_wav = vocoder.infer_waveform(spec)

    # Post-process the waveform
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    
    return generated_wav

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'form_submitted' in request.form:
        text = request.form['text']
        file = request.files.get('reference_audio')
        recorded_audio = request.form.get('recorded_audio')
        
        reference_audio_path = None
        
        if file:
            filename = secure_filename(file.filename)
            reference_audio_path = os.path.join("uploads", filename)
            file.save(reference_audio_path)
        elif recorded_audio:
            reference_audio_path = os.path.join("uploads", "recorded_audio.wav")
            audio_data = recorded_audio.split(",")[1]
            audio_bytes = base64.b64decode(audio_data)
            temp_audio_path = os.path.join("uploads", "temp_recorded_audio.webm")
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes)
            
            # Convert the recorded audio to WAV format
            audio = AudioSegment.from_file(temp_audio_path, format="webm")
            audio.export(reference_audio_path, format="wav")
            os.remove(temp_audio_path)
        
        if reference_audio_path:
            generated_wav = generate_audio(text, reference_audio_path)
            
            # Save the generated audio to a file
            output_path = "static/output.wav"
            sf.write(output_path, generated_wav, samplerate=synthesizer.sample_rate)
            return render_template('index.html', audio_file=output_path)
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
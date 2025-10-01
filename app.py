import gradio as gr
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import numpy as np
import json
import os

# Directory containing config files
CONFIG_DIR = "assets/Viet-SpeechT5-TTS-finetuning"

# Load all config.json files
def load_configs(directory):
    print(f"Searching for config files in: {directory}")
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []
    
    examples = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "config.json":
                file_path = os.path.join(root, file)
                print(f"Found config file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        if "input_text" in config and "voice" in config and "output_audio_path" in config:
                            examples.append([config["input_text"], config["voice"], os.path.join(root, config["output_audio_path"])])
                        else:
                            print(f"Skipping {file_path}: Missing required fields")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    print(f"Total examples loaded: {len(examples)}")
    return examples

# Load processor, model, and vocoder
print("Loading processor, model, and vocoder...")
try:
    processor = SpeechT5Processor.from_pretrained("danhtran2mind/Viet-SpeechT5-TTS-finetuning")
    model = SpeechT5ForTextToSpeech.from_pretrained("danhtran2mind/Viet-SpeechT5-TTS-finetuning")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Load speaker embeddings
print("Loading speaker embeddings...")
try:
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    print("Speaker embeddings loaded successfully.")
except Exception as e:
    print(f"Error loading embeddings: {e}")
    raise

def generate_speech(text, voice, output_path="output_speech.wav"):
    print(f"Generating speech for text: {text}, voice: {voice}, output: {output_path}")
    if not text or not voice:
        return None, "Please provide both text and voice selection."
    
    speaker_dict = {"male": 2000, "female": 7000}
    try:
        speaker_id = speaker_dict[voice.lower()]
        speaker_embedding = torch.tensor(embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)
        inputs = processor(text=text, return_tensors="pt")
        
        with torch.no_grad():
            speech = model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings=speaker_embedding,
                vocoder=vocoder,
                attention_mask=inputs.get("attention_mask")
            )
        
        sf.write(output_path, speech.numpy(), samplerate=16000)
        print(f"Audio saved to {output_path}")
        return output_path, None
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return None, f"Error generating speech: {str(e)}"

def load_existing_audio(text, voice, output_audio_path):
    print(f"Load: {output_audio_path}")
    if not output_audio_path:
        return text, voice, None, "Please select an existing audio file."
    return text, voice, output_audio_path, "Successfully Loaded Example Audio"

# Load examples
print("Loading examples...")
examples = load_configs(CONFIG_DIR)
if not examples:
    print("Warning: No examples loaded. Check CONFIG_DIR and config.json files.")

# Create Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Vietnamese Text-to-Speech")
    gr.Markdown("Generate speech from Vietnamese text using SpeechT5 or load existing audio from examples.")
    
    # Arrange components in a row
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Vietnamese Text", placeholder="Enter text here...", lines=5)
            voice_input = gr.Radio(choices=["Male", "Female"], label="Voice", value="Male")
            
        with gr.Column():
            output = gr.Audio(label="Generated Speech") 
            error_output = gr.Textbox(label="Error Message")
    
    # Button to generate speech
    gr.Button("Generate").click(
        fn=generate_speech,
        inputs=[text_input, voice_input],
        outputs=[output, error_output]
    )
    
    # Examples component
    gr.Examples(
        examples=examples,
        fn=load_existing_audio,
        inputs=[text_input, voice_input, output],
        outputs=[text_input, voice_input, output, error_output],
        label="Examples (Loads existing audio)"
    )

# Launch app
if __name__ == "__main__":
    print("Launching Gradio interface...")
    try:
        iface.launch()
        print("Gradio interface launched. Open your browser to http://localhost:7860")
    except Exception as e:
        print(f"Error launching Gradio interface: {e}")
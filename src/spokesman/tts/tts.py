from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

class TTS:
    def __init__(self):
        model_id = "gradium/phonon-40m-en"
        self.processor = AutoProcessor.from_pretrained(model_id, device_map="cuda")
        self.model = GradiumPhononTTSForConditionalGeneration.from_pretrained(model_id, device_map="cuda")

        self.voices =[
            {"name": "Maeve", "lang": "f-en-uk", "id": "6PWnV0Nq4wu7RVBT", "voice_path" : "", "available": False},
            {"name": "Freya", "lang": "f-en-uk", "id": "GgfEkEJtxZR7gnpy", "voice_path" : "", "available": False},
            {"name": "Elodie-Rose", "lang": "f-en-uk", "id": "y9bQqIDnyxwqK01t", "voice_path" : "", "available": False},
            {"name": "Alfie", "lang": "m-en-uk", "id": "X6NceqF285Gr04gb", "voice_path" : "", "available": False},
            {"name": "Hugh", "lang": "m-en-uk", "id": "YyKcxPG7iIgS3jsK", "voice_path" : "", "available": False},
            {"name": "Spencer", "lang": "m-en-uk", "id": "JOW0vchyswkQUMDS", "voice_path" : "", "available": False},
        ]

        self.download_all_voices()

        # Load first available voice as default
        default_voice = next((voice for voice in self.voices if voice["available"]), None)
        if default_voice:
            self.voice = load_file(default_voice["voice_path"])["speaker_wavs"][0].T.float()  # (num_frames, 512)
            print(f"Loaded default voice: {default_voice['name']} ({default_voice['lang']})") # debug
        else:
            raise ValueError("No available voices found.")

    def download_all_voices(self) -> None:
        for voice in self.voices:
            try : 
                voice_path = hf_hub_download("gradium/phonon-voices", f"voices/{voice['id']}.safetensors")
                voice['available'] = True
                voice['voice_path'] = voice_path
                print(f"Downloaded voice: {voice['name']} ({voice['lang']})") # debug
            except Exception as e:
                print(f"Failed to download voice: {voice['name']} ({voice['lang']}) - {e}") # debug
                continue

    def generate_audio(self, input_text):
        print("Start generating audio...")
        conversation = [
            {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": input_text}]},
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            processor_kwargs={"audio_prompt_latents": [self.voice]},
        ).to(self.model.device)

        outputs = self.model.generate(**inputs)
        audio = self.processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
        self.processor.save_audio(audio, "speech.wav")

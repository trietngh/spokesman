import spaces
from huggingface_hub import hf_hub_download
import numpy as np
from safetensors.torch import load_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

class TTS:
    def __init__(self):
        model_id = "gradium/phonon-40m-en"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = GradiumPhononTTSForConditionalGeneration.from_pretrained(model_id)
        self.sample_rate = self.processor.feature_extractor.sampling_rate

        self.voice_list =[
            {"name": "Maeve", "lang": "f-en-uk", "id": "6PWnV0Nq4wu7RVBT", "voice_path" : "", "available": False},
            {"name": "Freya", "lang": "f-en-uk", "id": "GgfEkEJtxZR7gnpy", "voice_path" : "", "available": False},
            {"name": "Elodie-Rose", "lang": "f-en-uk", "id": "y9bQqIDnyxwqK01t", "voice_path" : "", "available": False},
            {"name": "Alfie", "lang": "m-en-uk", "id": "X6NceqF285Gr04gb", "voice_path" : "", "available": False},
            {"name": "Hugh", "lang": "m-en-uk", "id": "YyKcxPG7iIgS3jsK", "voice_path" : "", "available": False},
            {"name": "Spencer", "lang": "m-en-uk", "id": "JOW0vchyswkQUMDS", "voice_path" : "", "available": False},
        ]

        self.download_all_voices()

        # Load first available voice as default
        self.current_voice_metadata = next((voice for voice in self.voice_list if voice["available"]), None)
        if self.current_voice_metadata:
            self.current_voice = load_file(self.current_voice_metadata["voice_path"])["speaker_wavs"][0].T.float()  # (num_frames, 512)
            print(f"Loaded default voice: {self.current_voice_metadata['name']} ({self.current_voice_metadata['lang']})") # debug
        else:
            raise ValueError("No available voices found.")

    def get_available_voice_names(self) -> list:
        return [voice["name"] for voice in self.voice_list if voice["available"]]

    def get_current_voice_name(self) -> str | None:
        return self.current_voice_metadata["name"] if self.current_voice_metadata else None

    def set_voice_by_name(self, voice_name: str) -> None:
        voice_metadata = next((voice for voice in self.voice_list if voice["name"] == voice_name and voice["available"]), None)
        if voice_metadata:
            self.current_voice_metadata = voice_metadata
            self.current_voice = load_file(voice_metadata["voice_path"])["speaker_wavs"][0].T.float()  # (num_frames, 512)
            print(f"Set voice to: {self.current_voice_metadata['name']} ({self.current_voice_metadata['lang']})") # debug
        else:
            raise ValueError(f"Voice '{voice_name}' not found or not available.")

    def download_all_voices(self) -> None:
        for voice in self.voice_list:
            try : 
                voice_path = hf_hub_download("gradium/phonon-voices", f"voices/{voice['id']}.safetensors")
                voice['available'] = True
                voice['voice_path'] = voice_path
                print(f"Downloaded voice: {voice['name']} ({voice['lang']})") # debug
            except Exception as e:
                print(f"Failed to download voice: {voice['name']} ({voice['lang']}) - {e}") # debug
                continue

    @spaces.GPU(duration=60)
    def generate_audio(self, input_text: str, voice_name: str | None = None) -> tuple[int, np.ndarray]:
        if not input_text or not input_text.strip():
            return self.sample_rate, np.zeros(0, dtype=np.float32)

        if voice_name:
            self.set_voice_by_name(voice_name)
        print(f"Start generating audio with voice {self.get_current_voice_name()}...") # debug
        conversation = [
            {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": input_text.strip()}]},
        ]

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = self.model.to(device)
            if hasattr(self.processor, "audio_tokenizer") and self.processor.audio_tokenizer is not None:
                self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(device)

            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                processor_kwargs={"audio_prompt_latents": [self.current_voice]},
            ).to(device)

            outputs = self.model.generate(**inputs)
            audio = self.processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
        except Exception as e:
            print(f"CUDA generation error ({e}); executing CPU fallback...") # debug
            self.model = self.model.to("cpu")
            if hasattr(self.processor, "audio_tokenizer") and self.processor.audio_tokenizer is not None:
                self.processor.audio_tokenizer = self.processor.audio_tokenizer.to("cpu")

            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                processor_kwargs={"audio_prompt_latents": [self.current_voice]},
            ).to("cpu")

            outputs = self.model.generate(**inputs)
            audio = self.processor.decode(outputs.audio_latents, outputs.audio_latents_mask)

        # Gradio Audio component expects (sample_rate, numpy_array)
        waveform = audio[0] if isinstance(audio, list) else audio
        audio_np = waveform.squeeze().cpu().float().numpy()

        return self.sample_rate, audio_np

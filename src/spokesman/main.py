from spokesman.tts import TTS
import gradio as gr

def spokesman():
    try:
        tts = TTS()
    except ValueError as e:
        print(f"Error initializing TTS: {e}")
        return
    
    available_voices = tts.get_available_voice_names()
    with gr.Blocks() as app:
        gr.HTML(
            f"""
            <h1 style='text-align: center;'> 🇬🇧 Spokesman 🇬🇧 </h1>
            <h3 style='text-align: center;'> Input your text and I will read it out loud. </h3>
            <p style='text-align: center;'> You can also select a voice from the dropdown menu. </p>
            """
        )

        with gr.Row():
            input_text = gr.Textbox(label="Input Text", placeholder="Type your text here...")

            voice = gr.Dropdown(
                label="Select a voice",
                choices=available_voices,
                value=available_voices[0],
            )

        with gr.Row():
            btn_gen = gr.Button("Generate Audio")
        with gr.Row():
            # output_text = gr.Textbox(label="Output Text", placeholder="The generated audio will be displayed here...")
            # Output audio component to play the generated audio
            output_audio = gr.Audio(label="Generated Audio")

        btn_gen.click(
            fn=tts.generate_audio,
            inputs=[input_text],
            outputs=[output_audio],
        )

        voice.change(
            fn=tts.set_voice_by_name,
            inputs=[voice],                    
        )

    app.launch()

if __name__ == "__main__":
    spokesman()
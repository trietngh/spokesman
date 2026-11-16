import gradio as gr

def generate_audio(input_text, voice):
    # Placeholder for audio generation logic
    # In a real implementation, you would call your TTS model here
    output_text = f"Generated audio for: '{input_text}' with voice '{voice}'"
    return output_text

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
            choices=[
                "en_1",
                "en_2",
                "en_3",
                "en_4",
                "en_5",
                "en_6",
                "en_7",
                "en_8",
                "en_9",
                "en_10",
            ],
            value="en_1",
        )

    with gr.Row():
        btn_gen = gr.Button("Generate Audio")
    with gr.Row():
        output_text = gr.Textbox(label="Output Text", placeholder="The generated audio will be displayed here...")

    btn_gen.click(
        fn=generate_audio,
        inputs=[input_text, voice],
        outputs=[output_text],
    )
    


app.launch()
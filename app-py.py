import streamlit as st
import torch
from PIL import Image
import os
from dotenv import load_dotenv
import google.generativeai as genai
from diffusers import StableDiffusionPipeline

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI API
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("GOOGLE_API_KEY is not set. Please set it in your environment.")
    st.stop()

@st.cache_resource
def load_text_model():
    model = genai.GenerativeModel("gemini-pro")
    return model

def generate_story_extension(user_input, model, temperature=0.85, max_output_tokens=1000):
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    response = model.generate_content(user_input, generation_config=generation_config)
    
    if response.parts:
        return ''.join(part.text for part in response.parts if hasattr(part, 'text'))
    else:
        return "No content generated."

@st.cache_resource
def load_image_model():
    model_id = "stabilityai/stable-diffusion-2"
    
    try:
        model = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float32,
            revision="fp16"
        )
        st.success(f"Successfully loaded the image generation model")
        return model
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {str(e)}")
        return None

def main():
    st.title("Interactive Storyteller")

    text_model = load_text_model()
    image_model = load_image_model()

    if image_model is None:
        st.error("Failed to load the image generation model. Please check the errors above and try again.")
        return
    
    characters = st.text_area("Enter the name of characters along with their descriptions")
    plot = st.text_area("Describe the plot in brief")
    theme = st.text_area("Provide a theme of the story along with the setting")

    if st.button("Generate Story"):
        user_input = f"Theme and setting: {theme} plot: {plot} characters: {characters}"
        prompt = f"Generate a complete story of at least 800 words based on input given by the user: \n {user_input}"
        
        with st.spinner("Generating initial story..."):
            base_story = generate_story_extension(prompt, text_model)
        
        st.subheader("Generated Story")
        st.write(base_story)

        st.session_state.story = base_story
        st.session_state.story_generated = True

    if 'story_generated' in st.session_state and st.session_state.story_generated:
        user_changed_input = st.text_input("If you want changes, type out the changes you want in the form of a simple prompt. Otherwise, leave blank to proceed.")
        
        if user_changed_input:
            with st.spinner("Updating story..."):
                prompt_change = f"Generate a new story from scratch of at least 800 words with reference to the previously generated story and based on changes instructed by the user: \n {user_changed_input}"
                updated_story = generate_story_extension(prompt_change, text_model)
            st.session_state.story = updated_story
            st.subheader("Updated Story")
            st.write(updated_story)

        if st.button("Generate Images"):
            with st.spinner("Generating story sequence and images..."):
                prompt_for_img_gen = f"Based on the story: {st.session_state.story}, add \n delimiters to separate the story into at least 10 pivotal parts (each part represents a different pivotal chapter of the story) and at max 20 parts, where each pivotal part gives an illustrative description of 30 to 40 words about that part such that an image can be generated from the part and fed into a text-to-image model to show progressive story. Try involving new settings or new characters in each part."
                
                prompt_corpus = generate_story_extension(prompt_for_img_gen, text_model)
                
                story_sequence = [x for x in prompt_corpus.split("\n") if 'Part' in x]

                st.subheader("Story Sequence")
                for part in story_sequence:
                    st.write(part)

                st.subheader("Generated Images")
                for i, part in enumerate(story_sequence):
                    image = image_model(part).images[0]
                    st.image(image, caption=f"Part {i+1}")
                    st.write(part)

if __name__ == "__main__":
    main()

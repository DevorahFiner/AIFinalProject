import subprocess
import sys

# Function to install a library
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary libraries
install("streamlit")
install("transformers")
install("gtts")

# Importing libraries after installation
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os

# Set up the Hugging Face model
MODEL_NAME = "MathLLMs/MathCoder-L-7B"  # Replace with your desired Hugging Face model

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

def generate_math_problem(level):
    prompt = f"Create a simple {level} math problem for a child."
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(inputs, max_length=50, num_return_sequences=1)
    problem = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return problem

def solve_math_problem(problem):
    prompt = f"Solve this math problem: {problem}"
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(inputs, max_length=50, num_return_sequences=1)
    solution = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return solution

def tutor(level):
    problem = generate_math_problem(level)
    solution = solve_math_problem(problem)
    return problem, solution

def text_to_speech(text):
    tts = gTTS(text)
    tts.save("output.mp3")
    return "output.mp3"

# Streamlit app
st.title("🧮 Math Tutor for Kids")
st.write("This app helps children learn math by generating and solving math problems.")

# Select difficulty level
level = st.selectbox(
    "Select the difficulty level of the math problem:",
    options=["easy", "medium", "hard"]
)

if st.button("Generate Math Problem"):
    problem, solution = tutor(level)
    
    # Display the math problem
    st.subheader("Math Problem")
    st.write(problem)
    
    # Display the solution
    st.subheader("Solution")
    st.write(solution)
    
    # Generate and display the audio solution
    audio_file = text_to_speech(solution)
    audio_bytes = open(audio_file, 'rb').read()
    st.audio(audio_bytes, format='audio/mp3')

# Instructions
st.markdown("### Instructions")
st.markdown("""
1. Select the difficulty level of the math problem.
2. Click 'Generate Math Problem' to see the problem and its solution.
3. The solution will also be available as audio.
""")


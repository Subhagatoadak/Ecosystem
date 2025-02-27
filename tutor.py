import streamlit as st
import requests
import os
from PIL import Image

# For image classification (placeholder)
from transformers import pipeline
from llm_service.llm_generator import generate_llm_response
##############################################
# Interview & Assessment Functionality
##############################################
def assess_answer(answer):
    """
    Uses LLM to evaluate an interview answer. The prompt asks for assessment on clarity,
    content, tonality, and provides improvement suggestions.
    """
    prompt = (
        "You are an experienced interviewer and coach. Evaluate the following answer for clarity, "
        "content, and tonality. Provide detailed feedback on strengths and improvement steps, including "
        "tips on how to improve tonality. Answer:\n\n"
        f"{answer}\n\n"
        "Please provide a concise evaluation and actionable improvement tips."
    )
    return generate_llm_response(prompt, provider="openai", model="gpt-4o", temperature=0.7)

##############################################
# Attire Analysis (Image Description)
##############################################
def analyze_formal_wear(image_file):
    """
    Uses an image classification pipeline to simulate analysis of whether the user is wearing formal attire.
    In production, replace or refine with a dedicated clothing detection model.
    """
    try:
        # Load a pre-trained image classification pipeline (this may download the model on first run)
        classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
        image = Image.open(image_file)
        predictions = classifier(image)
        # Look for keywords related to formal clothing
        formal_keywords = ["suit", "tuxedo", "blazer", "formal"]
        found = False
        for pred in predictions:
            label = pred["label"].lower()
            if any(keyword in label for keyword in formal_keywords):
                found = True
                break
        if found:
            return "The image analysis indicates that you are wearing formal attire. Great job!"
        else:
            return ("The image analysis suggests that you might not be wearing formal attire. "
                    "Consider dressing more formally for professional settings.")
    except Exception as e:
        return f"Error in analyzing image: {str(e)}"

##############################################
# Web Resource Search Functionality
##############################################
def search_web_resources(query):
    """
    Uses the LLM to simulate a web search for high-quality learning resources.
    """
    prompt = f"List some high-quality online resources (articles, tutorials, videos) to learn about {query}."
    response = generate_llm_response(prompt, provider="openai", model="gpt-4o", temperature=0.7)
    return response

##############################################
# Streamlit App Main Function
##############################################
def main():
    st.set_page_config(page_title="GenAI Tutor", layout="wide")
    
    # Inject custom CSS for an ultra-modern, colorful UI
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main {
            background-color: #f0f2f6;
        }
        h1, h2, h3 {
            color: #1a73e8;
        }
        .stButton>button {
            background-color: #1a73e8;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5em 1em;
            font-size: 16px;
        }
        .stTextInput>div>div>input {
            border: 2px solid #1a73e8;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("GenAI Tutor: Learn Python & Generative AI")
    
    # Sidebar Navigation for multiple modes
    app_mode = st.sidebar.selectbox("Choose the App Mode", [
        "Chatbot",
        "Web Resource Search",
        "GenAI Tutor Lessons",
        "Interview & Assessment",
        "Attire Analysis"
    ])
    
    ##############################################
    # Chatbot Mode
    ##############################################
    if app_mode == "Chatbot":
        st.header("Chatbot Interface")
        prompt = st.text_input("Enter your question or topic:")
        provider = st.selectbox("Choose LLM Provider", ["openai", "huggingface", "claude", "gemini"])
        model = st.text_input("Model", value="gpt-4o")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        if st.button("Submit"):
            with st.spinner("Generating response..."):
                response = generate_llm_response(prompt, provider=provider, model=model, temperature=temperature)
            st.markdown("**Response:**")
            st.write(response)
    
    ##############################################
    # Web Resource Search Mode
    ##############################################
    elif app_mode == "Web Resource Search":
        st.header("Web Resource Search")
        query = st.text_input("Enter a topic to search for resources:")
        if st.button("Search"):
            with st.spinner("Searching for resources..."):
                search_results = search_web_resources(query)
            st.markdown("**Search Results:**")
            st.write(search_results)
    
    ##############################################
    # GenAI Tutor Lessons Mode
    ##############################################
    elif app_mode == "GenAI Tutor Lessons":
        st.header("GenAI Tutor Lessons")
        lesson_topic = st.selectbox("Select a lesson topic", [
            "Introduction to Python",
            "Advanced Python Concepts",
            "Introduction to Generative AI",
            "Prompt Engineering",
            "Building Chatbots with Python"
        ])
        if st.button("Get Lesson"):
            with st.spinner("Generating lesson content..."):
                lesson_prompt = (
                    f"Provide a comprehensive lesson on {lesson_topic} that includes clear explanations, "
                    "code examples, and practical exercises. Write in a friendly and engaging tone."
                )
                lesson_content = generate_llm_response(lesson_prompt, provider="openai", model="gpt-4o", temperature=0.7)
            st.markdown("**Lesson Content:**")
            st.write(lesson_content)
    
    ##############################################
    # Interview & Assessment Mode
    ##############################################
    elif app_mode == "Interview & Assessment":
        st.header("Interview & Assessment")
        st.markdown("### Practice Interview")
        interview_question = st.text_area("Interview Question (or use a default one):", 
                                          value="Describe a challenging project you have worked on and how you overcame difficulties.")
        if st.button("Generate Interview Question"):
            st.info("Use the above question for your practice interview.")
        
        st.markdown("### Your Answer")
        user_answer = st.text_area("Type your answer here:")
        if st.button("Assess My Answer"):
            if user_answer.strip() == "":
                st.error("Please provide an answer before assessment.")
            else:
                with st.spinner("Evaluating your answer..."):
                    evaluation = assess_answer(user_answer)
                st.markdown("**Evaluation & Improvement Tips:**")
                st.write(evaluation)
    
    ##############################################
    # Attire Analysis Mode
    ##############################################
    elif app_mode == "Attire Analysis":
        st.header("Attire Analysis")
        st.markdown("Upload an image (e.g., from a video interview setup) to see if you're wearing formal attire.")
        image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if image_file is not None:
            with st.spinner("Analyzing image..."):
                result = analyze_formal_wear(image_file)
            st.markdown("**Analysis Result:**")
            st.write(result)
            st.image(Image.open(image_file), caption="Uploaded Image", use_column_width=True)

if __name__ == "__main__":
    main()
import os
import re
import pandas as pd
import streamlit as st
from transformers import pipeline  # pip install transformers
import spacy
from llm_service.llm_response import generate_llm_response  # Your LLM response generator module

############################
# 1. ADVANCED FACT EXTRACTION LIBRARY
############################

class FactExtractor:
    """
    A modular fact extractor that can use various NLP methods.
    Supported methods: 'regex', 'spacy', 'transformers', 'llm'.
    """
    def __init__(self, method='regex'):
        self.method = method.lower()
        if self.method == 'spacy':
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                st.error("Error loading spaCy model. Please run: python -m spacy download en_core_web_sm")
                raise e
        elif self.method == 'transformers':
            self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        elif self.method == 'llm':
            # For an LLM-based extractor, you might call your LLM service with a custom prompt.
            # For now, we simply fallback to regex.
            pass

    def extract_fact_value(self, text: str, fact_keyword: str):
        if self.method == 'regex':
            return self.extract_fact_value_regex(text, fact_keyword)
        elif self.method == 'spacy':
            return self.extract_fact_value_spacy(text, fact_keyword)
        elif self.method == 'transformers':
            return self.extract_fact_value_transformers(text, fact_keyword)
        elif self.method == 'llm':
            return self.extract_fact_value_llm(text, fact_keyword)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")

    def extract_fact_value_regex(self, text: str, fact_keyword: str):
        """
        Simple regex-based extraction.
        Looks for the fact keyword (e.g. "population") followed by a number.
        """
        pattern = re.compile(rf'{fact_keyword}\s*[:\-]?\s*.*?(\d+(?:\.\d+)?)', re.IGNORECASE)
        match = pattern.search(text)
        if match:
            try:
                value = float(match.group(1))
                start, end = match.span()
                context = text[max(0, start-50): end+50]
                return value, context.strip()
            except ValueError:
                return None, None
        return None, None

    def extract_fact_value_spacy(self, text: str, fact_keyword: str):
        """
        Uses spaCy to find sentences that mention the fact_keyword and then extracts the first numeric token.
        """
        doc = self.nlp(text)
        for sent in doc.sents:
            if fact_keyword.lower() in sent.text.lower():
                for token in sent:
                    if token.like_num:
                        try:
                            value = float(token.text.replace(",", ""))
                            return value, sent.text
                        except ValueError:
                            continue
        return None, None

    def extract_fact_value_transformers(self, text: str, fact_keyword: str):
        """
        Uses a Hugging Face transformers NER pipeline to extract entities near the fact_keyword.
        """
        results = self.ner_pipeline(text)
        idx = text.lower().find(fact_keyword.lower())
        if idx == -1:
            return None, None
        # Filter entities that are within 100 characters of the fact keyword occurrence.
        candidates = [r for r in results if abs(r['start'] - idx) < 100]
        for candidate in candidates:
            # Attempt to parse the entity as a number
            try:
                value = float(candidate['word'].replace(",", "").replace("##", ""))
                context_start = max(0, candidate['start'] - 50)
                context_end = candidate['end'] + 50
                context = text[context_start:context_end]
                return value, context
            except ValueError:
                continue
        return None, None

    def extract_fact_value_llm(self, text: str, fact_keyword: str):
        """
        Uses an LLM (via a custom prompt) to extract the numeric value for a fact.
        This is a placeholder; for now, it reverts to regex.
        """
        # In production, you might call generate_llm_response() with a prompt like:
        #   f"Extract the numeric value for {fact_keyword} from the following text: {text}"
        # And then parse the LLM output.
        return self.extract_fact_value_regex(text, fact_keyword)


############################
# 2. DATA & CONFIGURATION
############################

# Reference table: each row defines a fact keyword and its expected numeric value.
REFERENCE_DATA = {
    "Fact": ["population", "gdp", "area"],
    "Value": [331, 21175, 9834],  # e.g., USA: population (in millions), GDP (in billions USD), area (in thousand km²)
    "Unit": ["million", "billion USD", "thousand km²"]
}
df_reference = pd.DataFrame(REFERENCE_DATA)

# Relative tolerance for numeric comparison (e.g., allow 5% difference)
TOLERANCE = 0.05

def isclose_enough(val1, val2, tolerance=TOLERANCE):
    """
    Checks if two numeric values are within the given relative tolerance.
    """
    return abs(val1 - val2) / max(abs(val2), 1e-9) < tolerance

def evaluate_response_against_table(llm_text: str, df: pd.DataFrame, extraction_method='regex'):
    """
    For each fact (keyword) in the reference table, use the specified extraction method to:
      1. Extract the numeric value from the LLM response.
      2. Compare it to the reference value.
      3. Log any discrepancies or missing facts.
    Returns a DataFrame of error records.
    """
    extractor = FactExtractor(method=extraction_method)
    errors = []
    for _, row in df.iterrows():
        fact = row["Fact"]
        ref_value = row["Value"]
        extracted_value, context = extractor.extract_fact_value(llm_text, fact)
        if extracted_value is None:
            errors.append({
                "Fact": fact,
                "ExtractedValue": None,
                "ReferenceValue": ref_value,
                "ErrorType": "Not Found",
                "Context": f"Keyword '{fact}' not found or no numeric value extracted."
            })
        elif not isclose_enough(extracted_value, ref_value):
            errors.append({
                "Fact": fact,
                "ExtractedValue": extracted_value,
                "ReferenceValue": ref_value,
                "ErrorType": "Mismatch",
                "Context": context
            })
    return pd.DataFrame(errors)

############################
# 3. STREAMLIT DASHBOARD
############################

def main():
    st.title("Advanced LLM Fact Accuracy Evaluator")
    
    st.subheader("Reference Table")
    st.dataframe(df_reference)
    
    st.write("Either generate an LLM response using a prompt or paste an existing response below for evaluation.")
    
    # --- LLM Response Generation Section ---
    st.markdown("### Generate LLM Response")
    prompt = st.text_area("Enter prompt for LLM generation", height=100, 
                          value="Provide the latest population, GDP, and area figures for the USA.")
    provider = st.selectbox("LLM Provider", options=["openai", "huggingface", "claude", "gemini"], index=0)
    model = st.text_input("Model Name", value="gpt-4o")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
    
    if st.button("Generate Response"):
        with st.spinner("Generating response..."):
            generated_response = generate_llm_response(prompt, provider=provider, model=model, temperature=temperature)
        st.success("Response generated!")
        st.text_area("LLM Generated Response", value=generated_response, height=150)
    else:
        generated_response = ""
    
    # --- Selection for Extraction Method ---
    st.markdown("### Select Extraction Method")
    extraction_method = st.selectbox("Extraction Method", options=["regex", "spacy", "transformers", "llm"], index=0,
                                     help="Choose the NLP method for fact extraction.")
    
    # --- Evaluation Section ---
    st.markdown("### Evaluate LLM Response Against Table")
    st.write("Paste an LLM response (if not generated above) or use the generated response for evaluation.")
    llm_text_input = st.text_area("LLM Response for Evaluation", height=150, value=generated_response)
    
    if st.button("Evaluate"):
        if not llm_text_input.strip():
            st.warning("Please provide an LLM response to evaluate.")
        else:
            df_errors = evaluate_response_against_table(llm_text_input, df_reference, extraction_method=extraction_method)
            if df_errors.empty:
                st.success("No mismatches found! The LLM response accurately reflects the reference facts.")
            else:
                st.error("Errors or mismatches detected:")
                st.dataframe(df_errors)
                # Accumulate error log in session state
                if "error_log" not in st.session_state:
                    st.session_state["error_log"] = pd.DataFrame(columns=df_errors.columns)
                st.session_state["error_log"] = pd.concat([st.session_state["error_log"], df_errors], ignore_index=True)
                st.markdown("**Accumulated Error Log:**")
                st.dataframe(st.session_state["error_log"])

if __name__ == "__main__":
    main()

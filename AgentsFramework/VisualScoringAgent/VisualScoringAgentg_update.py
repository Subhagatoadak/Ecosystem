# Import required libraries and modules.
# These modules include:
# - llm_service for interacting with our LLM generator functions.
# - logging for debug and informational messages.
# - numpy for numerical operations.
# - random for generating random numbers and selections.
# - scipy.stats for performing statistical tests (e.g., t-test).
# - semopy for Structural Equation Modeling (SEM) analysis.
# - pandas for data manipulation.
# - networkx for building and analyzing social network graphs.

import os 
import sys
# Add the root directory to sys.path if not already there.
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from llm_service.llm_generator import generate_llm_response, generate_image_description,generate_llm_json
import logging
import numpy as np
import random
from scipy.stats import ttest_ind
from semopy import Model
import pandas as pd
import networkx as nx  # Used for building a realistic social network graph.
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from PIL import Image
import torch
import clip

# Set up basic logging configuration for debugging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Advanced Multi-Agent Manga Scoring Framework with Cognitive & Social Enhancements ---
class LLM_MangaScoringFramework:
    def __init__(self, guidelines=None, guidelines_prompt=None, scale=10, provider="openai", model="gpt-4o", temperature=0.7):
        """
        Initializes the scoring framework.
        
        Parameters:
          guidelines: Optional string containing manual scoring guidelines.
          guidelines_prompt: Optional prompt to generate guidelines via the LLM.
          scale: The scoring scale (default is 10).
          provider: LLM provider name.
          model: LLM model name.
          temperature: Sampling temperature for LLM responses.
          
        This constructor also sets up agent memory and social network structures.
        """
        self.scale = scale
        self.provider = provider
        self.model = model
        self.temperature = temperature

        # Initialize agent memory: dictionary to store feedback for each agent.
        self.agent_memory = {}
        # Initialize an empty social network. This will later be built using a network model.
        self.social_network = {}

        # If guidelines are provided manually, use them.
        if guidelines:
            self.guidelines = guidelines
            self.guidelines_generated = False
        # Else, if a guidelines prompt is provided, use LLM to generate guidelines.
        elif guidelines_prompt:
            try:
                self.guidelines = generate_llm_response(guidelines_prompt, provider=self.provider, model=self.model, temperature=self.temperature)
                self.guidelines_generated = True
            except Exception as e:
                # Log an error if guideline generation fails, and use default guidelines.
                logger.error("Failed to generate guidelines: %s", e)
                self.guidelines = f"Score art quality, character design, narrative conveyance, emotional impact, and cultural authenticity on a scale of 1 to {scale} based on clarity, creativity, consistency, and cultural resonance."
                self.guidelines_generated = False
        else:
            # Use default guidelines if nothing is provided.
            self.guidelines = f"Score art quality, character design, narrative conveyance, emotional impact, and cultural authenticity on a scale of 1 to {scale} based on clarity, creativity, consistency, and cultural resonance."
            self.guidelines_generated = False

    # --------------------------------------
    # Social Network Construction
    # --------------------------------------
    def build_social_network(self, num_agents):
        """
        Builds a social network graph for the given number of agents using a Barabási–Albert model.
        
        Parameters:
          num_agents: Total number of agents/users.
        
        Returns:
          A dictionary mapping each agent ID (1-indexed) to a list of connected agent IDs.
          
        The Barabási–Albert model is used to simulate real-world social network properties.
        """
        # Create a Barabási–Albert graph where each new node attaches to 2 existing nodes.
        G = nx.barabasi_albert_graph(num_agents, 2)
        network = {}
        for node in G.nodes():
            # Convert node IDs from 0-indexed (NetworkX default) to 1-indexed.
            network[node + 1] = [neighbor + 1 for neighbor in G.neighbors(node)]
        return network

    # --------------------------------------
    # Cognitive Enhancements & Social Methods
    # --------------------------------------
    def simulate_agent_emotional_state(self, agent_id):
        """
        Simulates and returns an emotional state for an agent.
        
        Parameters:
          agent_id: The ID of the agent (not used directly here, but can be extended).
        
        Returns:
          A randomly chosen emotional state from an extensive list.
        """
        emotions = [
            "happy", "neutral", "sad", "excited", "anxious", 
            "angry", "fearful", "content", "disappointed", "curious", 
            "frustrated", "elated", "depressed", "bored", "surprised", 
            "inspired", "confused", "hopeful", "lonely", "proud", 
            "embarrassed", "calm", "optimistic", "pessimistic"
        ]
        state = random.choice(emotions)
        return state

    def update_agent_memory(self, agent_id, feedback):
        """
        Updates the memory log for an agent with new feedback.
        
        Parameters:
          agent_id: The ID of the agent.
          feedback: The feedback or response to store.
          
        This method simulates memory by maintaining a list of past feedback for each agent.
        """
        if agent_id in self.agent_memory:
            self.agent_memory[agent_id].append(feedback)
        else:
            self.agent_memory[agent_id] = [feedback]

    def simulate_social_interactions(self, responses):
        """
        Simulates social influence among agents based on the pre-built social network.
        
        Parameters:
          responses: A list of responses from the agents.
        
        Returns:
          A new list of responses adjusted with a social influence factor.
          
        Each agent's response is adjusted by appending an "influence factor" computed
        as 0.1 multiplied by the number of neighbors (connections) the agent has.
        """
        adjusted_responses = []
        for i, response in enumerate(responses, start=1):
            # Get the list of connected neighbors for agent i.
            neighbors = self.social_network.get(i, [])
            # Calculate influence factor (example: 0.1 per neighbor).
            influence_factor = 0.1 * len(neighbors)
            # Append a note indicating the social influence adjustment.
            adjusted_response = response + f"\n[Social Influence Adjustment Factor: {influence_factor:.2f}]"
            adjusted_responses.append(adjusted_response)
        return adjusted_responses

    # --------------------------------------
    # Existing Methods (with Enhancements)
    # --------------------------------------
    def generate_questionnaire(self, num_questions=5, user_prompt=None, expert_context=None, questionnaire_manual=None):
        """
        Generates a questionnaire for evaluating manga visuals.
        
        Parameters:
          num_questions: Number of questions to generate.
          user_prompt: Additional prompt text provided by the user.
          expert_context: Additional expert context to guide question generation.
          questionnaire_manual: If provided, bypass generation and use this questionnaire.
        
        Returns:
          A numbered list of questions as a string.
          
        This function adds environmental context (like current cultural trends)
        to make the questions more context-aware.
        """
        if questionnaire_manual:
            self.questionnaire_generated = False
            return questionnaire_manual

        # Use default empty strings if no prompt or context is provided.
        user_prompt = user_prompt if user_prompt else ""
        expert_context = expert_context if expert_context else ""
        # Environmental context to consider external influences.
        environmental_context = "Consider current cultural trends and the time-of-day which might affect visual perception."
        prompt = (
            f"Strictly generate {num_questions} detailed questions to evaluate manga visuals. Focus on aspects such as art quality, character design, "
            f"narrative impact, emotional tone, and cultural authenticity. Expert context: {expert_context}. User prompt: {user_prompt}. {environmental_context} "
            f"Return the questions as a numbered list."
        )
        logger.info("Generating questionnaire with prompt: %s", prompt)
        try:
            questionnaire = generate_llm_response(prompt, provider=self.provider, model=self.model, temperature=self.temperature)
            self.questionnaire_generated = True
        except Exception as e:
            logger.error("Failed to generate questionnaire: %s", e)
            questionnaire = ""
            self.questionnaire_generated = False
        return questionnaire

    def generate_audience_personality(self, audience_index, personality_prompt):
        """
        Generates a detailed personality profile for an audience member.
        
        Parameters:
          audience_index: The index/ID of the audience member.
          personality_prompt: A prompt to guide the generation of the personality profile.
        
        Returns:
          A personality profile string.
          
        The profile includes aspects of psychological traits, cultural background,
        and the current emotional state. It also considers any memory stored for the agent.
        """
        # Get current emotional state for the agent.
        current_emotion = self.simulate_agent_emotional_state(audience_index)
        # Enhance the prompt with the current emotional state and memory considerations.
        enhanced_prompt = (
            f"Using insights from personality psychology (e.g., the Big Five) and cultural theories (e.g., Hofstede's dimensions), "
            f"generate a detailed personality profile for audience member {audience_index}. Include psychological traits (openness, conscientiousness, extraversion, "
            f"agreeableness, neuroticism), cultural background, and influences. Also, incorporate the agent's current emotional state: '{current_emotion}', "
            f"and consider any previous feedback stored in memory. Use the following prompt for further guidance: {personality_prompt}"
        )
        try:
            personality = generate_llm_response(enhanced_prompt, provider=self.provider, model=self.model, temperature=self.temperature)
        except Exception as e:
            logger.error("Failed to generate audience personality for member %d: %s", audience_index, e)
            personality = (
                f"Audience member {audience_index} with a balanced psychological profile, diverse cultural background, and currently feeling {current_emotion}."
            )
        return personality

    def extract_image_features(self, image_path):
        """
        Extracts visual features from the image using the CLIP model and returns a textual summary.
        
        Parameters:
        image_path: Path to the image file.
        
        Returns:
        A string summarizing a few values from the image embedding.
        """
        # Use CUDA if available.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the pre-trained CLIP model and preprocessing pipeline.
        model, preprocess = clip.load("ViT-B/32", device=device)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to open image %s: %s", image_path, e)
            return "Image features unavailable."
        
        # Preprocess and encode the image.
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        
        # Convert the embedding into a simple textual summary.
        # For demonstration, we return the first 5 values of the embedding.
        feature_summary = f"Image Features (first 5 values): {image_features[0][:5].cpu().numpy().tolist()}"
        return feature_summary
    
    def simulate_audience_responses(self, image_path, questionnaire, num_profiles=5, audience_personalities=None, audience_personality_prompt=None):
        """
        Simulates responses from multiple audience members, incorporating both text and image features.
        
        Parameters:
        image_path: Path to the manga visual image.
        questionnaire: The questionnaire text for evaluating the image.
        num_profiles: Number of audience members to simulate.
        audience_personalities: Optional list of pre-generated personality profiles.
        audience_personality_prompt: Prompt for generating personality profiles if not provided.
        
        Returns:
        A tuple containing:
            - A list of simulated audience responses.
            - A list of generated personality profiles.
        
        The function builds a realistic social network, extracts both text-based image descriptions and visual features,
        and then combines these with personality prompts to generate comprehensive evaluation responses.
        """
        # Build the social network for the given number of agents.
        self.social_network = self.build_social_network(num_profiles)
        
        # Extract visual features from the image.
        image_feature_summary = self.extract_image_features(image_path)
        
        responses = []
        generated_personalities = []
        for i in range(1, num_profiles + 1):
            # Create a description prompt that includes cultural context and image features.
            description_prompt = (
                "Please provide a detailed description of this manga visual. "
                "Describe the scene, characters, action, emotions, and any cultural elements depicted. "
                "Also, take into account the current cultural context and trends. "
                "Consider the following evaluation questions when describing the image:\n"
                f"{questionnaire}\n"
                "Additional Visual Features: " + image_feature_summary
            )
            try:
                # Generate an image description using LLM.
                tailored_description = generate_image_description(
                    image_path,
                    description_prompt,
                    provider=self.provider,
                    model="gpt-4o-mini",
                    temperature=self.temperature
                )
            except Exception as e:
                logger.error("Failed to generate image description for profile %d: %s", i, e)
                tailored_description = "Description unavailable due to an error."
            
            # Determine the personality for the agent.
            if audience_personalities is not None and len(audience_personalities) >= i:
                personality = audience_personalities[i - 1]
            elif audience_personality_prompt is not None:
                personality = self.generate_audience_personality(i, audience_personality_prompt)
            else:
                # Default personality if nothing is provided.
                personality = f"audience member {i} with a balanced psychological profile, diverse cultural background, and insight on cultural narratives"
            generated_personalities.append(personality)
            
            # Construct the evaluation prompt that asks for both numeric scores and qualitative feedback.
            eval_prompt = (
                f"Based on the following tailored manga visual description:\n'{tailored_description}'\n\n"
                f"and the following questionnaire:\n{questionnaire}\n\n"
                f"Assume the role of {personality} and answer each question using a 1 to {self.scale} scale "
                f"(where 1 is very poor and {self.scale} is excellent). For each question, provide a brief justification for your score. "
                f"Also, include qualitative feedback on emotional tone, cultural relevance, and any personal reflections influenced by previous experiences.\n"
                f"Format your answer as a numbered list, with each line as: 'Question <number>: Score - Explanation'."
            )
            try:
                # Generate the audience response using LLM.
                response = generate_llm_response(eval_prompt, provider=self.provider, model=self.model, temperature=self.temperature)
            except Exception as e:
                logger.error("Failed to generate response for audience member %d: %s", i, e)
                response = f"Response unavailable for audience member {i}."
            responses.append(response)
            # Update the agent's memory with the generated response.
            self.update_agent_memory(i, response)
        
        # Apply social influence adjustments based on the built social network.
        responses = self.simulate_social_interactions(responses)
        
        return responses, generated_personalities

    def enforce_guidelines(self, responses):
        """
        Ensures that each audience response adheres to the scoring guidelines.
        
        Parameters:
          responses: The list of audience responses.
        
        Returns:
          Validated and potentially corrected responses.
          
        This method uses an LLM call to check that scores fall within the expected range and that justifications meet the guidelines.
        """
        prompt = (
            f"Review the following simulated audience responses:\n{responses}\n\n"
            f"Ensure that each score is between 1 and {self.scale} and that the justifications align with the following guidelines: {self.guidelines}.\n"
            f"If any score is out of range or the justification does not follow the guidelines, provide corrections. Otherwise, confirm that the responses are valid.\n"
            f"Return the validated responses."
        )
        try:
            validated_responses = generate_llm_response(prompt, provider=self.provider, model=self.model, temperature=self.temperature)
        except Exception as e:
            logger.error("Failed to enforce guidelines: %s", e)
            validated_responses = responses
        return validated_responses

    def aggregate_scores(self, responses):
        """
        Aggregates numeric scores from the validated responses.
        
        Parameters:
          responses: The list of validated audience responses.
        
        Returns:
          The final aggregated score (normalized out of 100) as a string.
          
        This function uses an LLM call to extract numeric scores from responses and compute the average.
        """
        prompt = (
            f"Given the following validated audience responses (each containing scores in the format 'Question <number>: Score - Explanation'):\n{responses}\n\n"
            f"Extract all the numeric scores and compute the average score for the manga visual. Return only the final aggregated score. The final score should be normalized to be out of 100."
        )
        try:
            final_score = generate_llm_response(prompt, provider=self.provider, model=self.model, temperature=self.temperature)
        except Exception as e:
            logger.error("Failed to aggregate scores: %s", e)
            final_score = "Score unavailable"
        return final_score

    # ------------------- Advanced Analytics, Cultural Aspects & Decision Modeling -------------------
    def extract_criteria_scores(self, responses):
        """
        Extracts scores for each evaluation criterion from responses in a production setting.
        
        Parameters:
        responses: A list of dictionaries. Each dictionary is expected to contain numeric ratings
                    for a set of evaluation criteria.
        
        Returns:
        A dictionary mapping each evaluation criterion to its mean score computed across all responses.
        If a criterion is not present in the responses, its value will be None.
        
        The list of criteria is extensive and covers artistic, narrative, cultural, and brand health metrics.
        """
        # Define an exhaustive list of required evaluation criteria.
        required_cols = [
            "art_quality",             # Overall aesthetic quality and craftsmanship.
            "character_design",        # Effectiveness in character portrayal.
            "narrative_impact",        # Impact of the narrative on the viewer.
            "emotional_impact",        # Emotional response elicited.
            "cultural_impact",         # Integration and authenticity of cultural elements.
            "composition_balance",     # Balance of visual elements.
            "color_harmony",           # Cohesion of color usage.
            "typography",              # Quality and integration of text.
            "gestalt_principles",      # Coherence of elements based on Gestalt theory.
            "visual_impact",           # Immediate visual appeal.
            "innovation",              # Novelty and creativity.
            "clarity",                 # Clarity in visual communication.
            "accessibility",           # Ease of understanding by diverse audiences.
            "narrative_structure",     # Logical flow of the narrative.
            "symbolism",               # Use of symbolic elements.
            "creativity",              # Overall creative expression.
            "context_relevance",       # Fit within the current cultural or market context.
            "soft_power",              # Ability to influence perceptions through culture.
            "foreign_influence",       # Presence of global or international trends.
            "local_mindset",           # Alignment with local cultural values.
            "cultural_identity",       # Reinforcement of local cultural identity.
            "cultural_spill",          # Degree of foreign culture integration.
            "brand_recall",            # Ease with which the brand is remembered.
            "brand_perception",        # Overall consumer perception of the brand.
            "brand_preference",        # Preference for the brand over competitors.
            "brand_equity",            # Overall strength and market value of the brand.
            "brand_consistency",       # Consistency of the brand messaging.
            "advertising_effectiveness",  # Effectiveness of the ad in communicating its message.
            "consumer_trust",          # Level of trust the brand engenders.
            "consumer_loyalty",        # Likelihood of repeat engagement.
            "social_media_engagement", # Engagement metrics from social media.
            "aesthetic_pleasure",      # Subjective pleasure derived from the visual.
            "emotional_resonance",     # Depth of the emotional connection.
            "engagement",              # Overall audience engagement.
            "message_clarity",         # Clarity of the brand's message.
            "empathy",                 # Ability to evoke empathetic responses.
            "sustainability"           # Perceived ethical appeal or sustainability.
        ]
        
        # Convert the list of response dictionaries to a pandas DataFrame.
        df = pd.DataFrame(responses)
        
        # Initialize a dictionary to store mean scores for each criterion.
        criteria_scores = {}
        
        # Iterate through each required column.
        for col in required_cols:
            if col in df.columns:
                try:
                    # Convert column values to numeric, coercing errors into NaN.
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Calculate the mean, ignoring NaN values.
                    criteria_scores[col] = df[col].dropna().mean()
                except Exception as e:
                    logger.error("Error computing mean for column %s: %s", col, e)
                    criteria_scores[col] = None
            else:
                # If the column is not present in the DataFrame, assign None.
                criteria_scores[col] = None

        return criteria_scores


    def apply_mcda_ahp(self, scores, criteria_weights):
        """
        Computes a weighted aggregated score using an Analytic Hierarchy Process (AHP)-like approach.
        
        Parameters:
          scores: Dictionary containing scores for each evaluation criterion.
          criteria_weights: Dictionary containing weights for each criterion.
        
        Returns:
          The MCDA/AHP weighted score as a float.
          
        The score is computed as the weighted average of the criteria scores.
        """
        total_weight = sum(criteria_weights.values())
        weighted_sum = 0
        for criterion in criteria_weights:
            # Use 0 if a score is missing.
            score = scores.get(criterion, 0)
            if score is None:
                score = 0
            weighted_sum += score * criteria_weights[criterion]
        weighted_score = weighted_sum / total_weight if total_weight != 0 else 0
        return weighted_score
    
    def apply_bayesian_updating(self, aggregated_score, prior_score=None, confidence=0.5):
        """
        Applies Bayesian updating to refine the aggregated score.
        
        Parameters:
          aggregated_score: The newly computed aggregated score.
          prior_score: The prior aggregated score (if available).
          confidence: The weight for the new score versus the prior score.
        
        Returns:
          The updated aggregated score as a float.
          
        This method blends the new score with the prior, based on the specified confidence level.
        """
        try:
            agg = float(aggregated_score)
        except (ValueError, TypeError):
            agg = 0.0
        if prior_score is not None:
            updated_score = confidence * agg + (1 - confidence) * prior_score
        else:
            updated_score = agg
        return updated_score

    def validate_with_sem(self, responses):
        """
        Performs Structural Equation Modeling (SEM) to validate latent constructs.
        
        Parameters:
        responses: A list of dictionaries with numeric ratings for a comprehensive set of evaluation criteria.
        
        The exhaustive list of required columns includes:
        - Traditional Artistic and Narrative Quality: art_quality, character_design, narrative_impact, emotional_impact, cultural_impact
        - Visual Graphic Theory Parameters: composition_balance, color_harmony, typography, gestalt_principles, visual_impact, innovation, clarity, accessibility
        - Narrative and Literary Elements: narrative_structure, symbolism, creativity, context_relevance
        - Cultural Influence and Globalization: soft_power, foreign_influence, local_mindset, cultural_identity, cultural_spill
        - Brand Health and Consumer Perception: brand_recall, brand_perception, brand_preference, brand_equity, brand_consistency, advertising_effectiveness, consumer_trust, consumer_loyalty, social_media_engagement
        - Additional Emotional and Experiential Metrics: aesthetic_pleasure, emotional_resonance, engagement, message_clarity, empathy, sustainability
        
        The updated SEM model specification is defined as:
        visual_quality         =~ art_quality + character_design + composition_balance + color_harmony + visual_impact
        narrative_quality      =~ narrative_impact + narrative_structure + symbolism + creativity + context_relevance
        cultural_authenticity  =~ cultural_impact + soft_power + foreign_influence + local_mindset + cultural_identity + cultural_spill
        brand_health           =~ brand_recall + brand_perception + brand_preference + brand_equity + brand_consistency + advertising_effectiveness + consumer_trust + consumer_loyalty + social_media_engagement
        emotional_experience   =~ emotional_impact + aesthetic_pleasure + emotional_resonance + engagement + message_clarity + empathy + sustainability
        
        Returns:
        A dictionary containing fit indices and parameter estimates from the SEM analysis.
        
        This analysis helps validate whether the observed variables load well onto the latent constructs.
        """
        try:
            # Convert the list of response dictionaries into a DataFrame.
            data = pd.DataFrame(responses)
            
            # Define the exhaustive list of required columns.
            required_cols = [
                "art_quality", "character_design", "narrative_impact", "emotional_impact", "cultural_impact",
                "composition_balance", "color_harmony", "typography", "gestalt_principles", "visual_impact",
                "innovation", "clarity", "accessibility",
                "narrative_structure", "symbolism", "creativity", "context_relevance",
                "soft_power", "foreign_influence", "local_mindset", "cultural_identity", "cultural_spill",
                "brand_recall", "brand_perception", "brand_preference", "brand_equity", "brand_consistency",
                "advertising_effectiveness", "consumer_trust", "consumer_loyalty", "social_media_engagement",
                "aesthetic_pleasure", "emotional_resonance", "engagement", "message_clarity", "empathy", "sustainability"
            ]
            
            # Verify that all required columns are present in the data.
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError("Data is missing one or more required columns: " + ", ".join(missing_cols))
            
            # Optionally, convert all required columns to numeric, coercing non-numeric entries to NaN.
            for col in required_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Define the SEM model specification with multiple latent constructs.
            model_desc = """
            visual_quality         =~ art_quality + character_design + composition_balance + color_harmony + visual_impact
            narrative_quality      =~ narrative_impact + narrative_structure + symbolism + creativity + context_relevance
            cultural_authenticity  =~ cultural_impact + soft_power + foreign_influence + local_mindset + cultural_identity + cultural_spill
            brand_health           =~ brand_recall + brand_perception + brand_preference + brand_equity + brand_consistency + advertising_effectiveness + consumer_trust + consumer_loyalty + social_media_engagement
            emotional_experience   =~ emotional_impact + aesthetic_pleasure + emotional_resonance + engagement + message_clarity + empathy + sustainability
            """
            
            # Create the SEM model.
            sem_model = Model(model_desc)
            # Fit the model using the data.
            sem_model.fit(data)
            # Calculate model fit statistics.
            stats = sem_model.calc_stats()
            # Retrieve parameter estimates.
            estimates = sem_model.inspect()
            sem_results = {
                "fit_indices": stats,
                "parameter_estimates": estimates.to_dict(orient="records")
            }
        except Exception as e:
            logger.error("SEM analysis failed: %s", e)
            sem_results = {"error": str(e)}
        
        return sem_results


    def advanced_analytics_pipeline(self, responses, aggregated_score, prior_score=None, criteria_weights=None):
        """
        Integrates Bayesian updating, SEM validation, and MCDA (via AHP) including cultural criteria.
        
        Parameters:
        responses: A list of responses that can be parsed into a DataFrame containing numeric ratings.
                    The DataFrame is expected to have an extensive set of columns such as:
                    art_quality, character_design, narrative_impact, emotional_impact, cultural_impact,
                    composition_balance, color_harmony, typography, gestalt_principles, visual_impact,
                    innovation, clarity, accessibility, narrative_structure, symbolism, creativity,
                    context_relevance, soft_power, foreign_influence, local_mindset, cultural_identity,
                    cultural_spill, brand_recall, brand_perception, brand_preference, brand_equity,
                    brand_consistency, advertising_effectiveness, consumer_trust, consumer_loyalty,
                    social_media_engagement, aesthetic_pleasure, emotional_resonance, engagement,
                    message_clarity, empathy, sustainability.
        aggregated_score: The initial aggregated score.
        prior_score: Optional prior aggregated score for Bayesian updating.
        criteria_weights: Optional dictionary specifying weights for each evaluation criterion.
                            If not provided, a default dictionary for a selected subset of criteria will be used.
        
        Returns:
        A dictionary containing:
            - initial_aggregated_score: The original aggregated score.
            - mcda_weighted_score: The weighted score computed via MCDA/AHP.
            - bayesian_updated_score: The aggregated score updated using Bayesian methods.
            - sem_validation: Results from the SEM analysis.
            
        The function computes the mean score for each criterion present in criteria_weights (or a default set)
        from the responses DataFrame, then computes a weighted average (MCDA), applies Bayesian updating, and finally
        performs SEM validation on the full dataset.
        """
        # Check if responses are in the expected dictionary format.
        if not responses or not isinstance(responses[0], dict):
            logger.warning("Responses are not in expected dictionary format for advanced analytics. Skipping advanced analytics pipeline.")
            try:
                agg_score = float(aggregated_score)
            except:
                agg_score = 0.0
            return {
                "initial_aggregated_score": agg_score,
                "mcda_weighted_score": agg_score,
                "bayesian_updated_score": self.apply_bayesian_updating(agg_score, prior_score, confidence=0.6),
                "sem_validation": {"error": "Responses not structured for SEM analysis."}
            }
        
        try:
            df = pd.DataFrame(responses)
        except Exception as e:
            logger.error("Failed to convert responses to DataFrame: %s", e)
            df = pd.DataFrame()
        
        if criteria_weights is None:
            criteria_weights = {
                "art_quality": 0.15,
                "character_design": 0.10,
                "narrative_impact": 0.10,
                "emotional_impact": 0.10,
                "cultural_impact": 0.10,
                "composition_balance": 0.05,
                "color_harmony": 0.05,
                "visual_impact": 0.05,
                "innovation": 0.05,
                "narrative_structure": 0.05,
                "symbolism": 0.05,
                "soft_power": 0.03,
                "brand_equity": 0.03,
                "consumer_trust": 0.03,
                "aesthetic_pleasure": 0.03
            }
        
        criteria_scores = {}
        for criterion in criteria_weights.keys():
            if criterion in df.columns:
                try:
                    df[criterion] = pd.to_numeric(df[criterion], errors='coerce')
                    criteria_scores[criterion] = df[criterion].dropna().mean()
                except Exception as e:
                    logger.error("Error computing mean for %s: %s", criterion, e)
                    criteria_scores[criterion] = 0
            else:
                logger.warning("Criterion %s not found in responses; setting score to 0", criterion)
                criteria_scores[criterion] = 0

        mcda_score = self.apply_mcda_ahp(criteria_scores, criteria_weights)
        try:
            agg_score = float(aggregated_score)
        except (ValueError, TypeError):
            agg_score = 0.0
        updated_score = self.apply_bayesian_updating(agg_score, prior_score, confidence=0.6)
        sem_validation = self.validate_with_sem(responses)
        
        return {
            "initial_aggregated_score": agg_score,
            "mcda_weighted_score": mcda_score,
            "bayesian_updated_score": updated_score,
            "sem_validation": sem_validation
        }

    def explain_process(self, responses, final_score):
        """
        Provides a detailed explanation of how the final score was derived.
        
        Parameters:
          responses: The list of audience responses.
          final_score: The final aggregated score.
        
        Returns:
          A detailed explanation string generated by an LLM.
          
        This explanation includes how each question contributed to the final score,
        corrections applied during guideline enforcement, and the integration of advanced analytics.
        """
        prompt = (
            f"Explain in detail how the final aggregated score of {final_score} was derived from the following audience responses:\n{responses}\n\n"
            f"Include how each question and its corresponding score contributed to the final score, any corrections made during guideline enforcement, "
            f"and highlight key insights from the process. Specifically, describe how Bayesian updating, SEM validation, and MCDA (via AHP) were integrated "
            f"to handle uncertainty, validate latent constructs (including cultural authenticity), and assign weights to evaluation criteria. Provide a detailed explanation for the final score."
        )
        try:
            explanation = generate_llm_response(prompt, provider=self.provider, model=self.model, temperature=self.temperature)
        except Exception as e:
            logger.error("Failed to explain the process: %s", e)
            explanation = "Explanation unavailable due to an error."
        return explanation

    # ------------------- Additional Enhancements for A/B Testing and Content Analysis -------------------
    def analyze_qualitative_feedback(self, responses):
        """
        Analyzes qualitative feedback from audience responses.
        
        Parameters:
          responses: The list of audience responses containing qualitative feedback.
        
        Returns:
          A concise summary of key themes, sentiments, and tone extracted from the responses.
          
        This function uses an LLM to perform sentiment and thematic analysis on the responses.
        """
        prompt = (
            f"Analyze the following qualitative feedback extracted from audience responses and summarize key themes, sentiments, and tone:\n{responses}\n"
            "Provide a concise summary of the qualitative insights."
        )
        try:
            qualitative_summary = generate_llm_response(prompt, provider=self.provider, model=self.model, temperature=self.temperature)
        except Exception as e:
            logger.error("Failed to analyze qualitative feedback: %s", e)
            qualitative_summary = "Qualitative analysis unavailable due to an error."
        return qualitative_summary

    def compute_inter_rater_reliability(self, responses):
        """
        Computes Cronbach's alpha based on numeric scores from multiple raters.
        
        Parameters:
          responses: A list of lists or a 2D numpy array of shape (n_raters, n_items), where each inner list contains numeric scores.
        
        Returns:
          Cronbach's alpha as a float rounded to two decimals.
          
        This metric evaluates the consistency of the ratings across different raters.
        """
        # Convert responses to a numpy array.
        data = np.array(responses)
        if data.ndim != 2:
            raise ValueError("Responses must be a 2D array (or list of lists) of numeric scores.")
        
        n_items = data.shape[1]
        # Compute the variance for each item using sample variance.
        item_variances = data.var(axis=0, ddof=1)
        # Sum the scores for each rater.
        total_scores = data.sum(axis=1)
        # Compute the total variance across raters.
        total_variance = total_scores.var(ddof=1)
        
        # Prevent division by zero.
        if total_variance == 0:
            return 0.0
        
        # Compute Cronbach's alpha using the standard formula.
        alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
        return round(alpha, 2)

    def perform_statistical_test(self, scores_A, scores_B):
        """
        Performs a two-sample t-test comparing aggregated scores from two groups.
        
        Parameters:
          scores_A: A list or array of aggregated scores for variant A.
          scores_B: A list or array of aggregated scores for variant B.
        
        Returns:
          The p-value from the t-test as a float rounded to three decimals.
          
        This test checks whether there is a statistically significant difference between the two variants.
        """
        scores_A = np.array(scores_A)
        scores_B = np.array(scores_B)
        t_stat, p_value = ttest_ind(scores_A, scores_B, equal_var=False)
        return round(p_value, 3)

    def segment_responses_by_demographic(self, personalities):
        """
        Segments audience personalities into demographic or psychographic groups using text embeddings and clustering.

        Parameters:
        personalities: A list of personality profile strings.

        Returns:
        A dictionary mapping cluster labels (e.g., "Segment 0", "Segment 1", etc.)
        to lists of personality profiles that belong to that segment.

        Production-Ready Explanation:
        - This function uses a pre-trained SentenceTransformer model (e.g., 'all-MiniLM-L6-v2')
            to convert each personality profile into a dense vector (embedding).
        - KMeans clustering is then applied to these embeddings to group profiles into clusters
            based on semantic similarity.
        - The number of clusters (num_clusters) is set to 2 by default but can be adjusted or
            dynamically determined in a production system.
        - This method provides a more robust and data-driven segmentation compared to rule-based approaches.
        """
        # Initialize the pre-trained sentence transformer model.
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Compute embeddings for each personality profile.
        embeddings = model.encode(personalities)
        
        # Define the number of clusters; in production, this may be tuned or passed as a parameter.
        num_clusters = 2
        
        # Initialize and fit the KMeans clustering model on the embeddings.
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Organize the personality profiles into clusters.
        segments = {}
        for cluster_id in range(num_clusters):
            segment_label = f"Segment {cluster_id}"
            segments[segment_label] = [
                personality for personality, label in zip(personalities, labels) if label == cluster_id
            ]
        
        return segments

    def display_dashboard(self, analytics_results, qualitative_summary, reliability, p_value, segments):
        """
        Production-ready display dashboard function using the Rich library.
        
        Parameters:
        analytics_results: Dictionary containing scores from the advanced analytics pipeline.
        qualitative_summary: Summary of qualitative feedback.
        reliability: Inter-rater reliability metric.
        p_value: p-value from the statistical test comparing variants.
        segments: Dictionary with demographic/psychographic segments of the audience.
        
        This function uses Rich to create a well-formatted dashboard with:
        - A table for key numerical metrics.
        - Panels for SEM validation results, qualitative feedback, and audience segments.
        """
        # Initialize the rich console.
        console = Console()
        
        # Create a table for the key numeric metrics.
        table = Table(title="Analytics Dashboard", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=30)
        table.add_column("Value", justify="right", style="cyan")
        
        # Add rows for each key metric.
        table.add_row("Initial Aggregated Score", str(analytics_results.get('initial_aggregated_score', 'N/A')))
        table.add_row("MCDA Weighted Score", str(analytics_results.get('mcda_weighted_score', 'N/A')))
        table.add_row("Bayesian Updated Score", str(analytics_results.get('bayesian_updated_score', 'N/A')))
        table.add_row("Inter-Rater Reliability", str(reliability))
        table.add_row("Statistical Test p-value", str(p_value))
        
        # Create a panel for the SEM validation results.
        sem_results = analytics_results.get('sem_validation', {})
        sem_panel = Panel(
            str(sem_results),
            title="SEM Validation Results",
            subtitle="Fit Indices and Parameter Estimates",
            style="green"
        )
        
        # Create a panel for the qualitative feedback summary.
        qual_panel = Panel(
            qualitative_summary,
            title="Qualitative Feedback Summary",
            style="blue"
        )
        
        # Format audience segments as a multi-line string.
        segments_str = "\n".join([f"[bold]{segment}:[/bold] {', '.join(personas)}" for segment, personas in segments.items()])
        segments_panel = Panel(
            segments_str,
            title="Audience Segments",
            style="yellow"
        )
        
        # Display the table and panels.
        console.print(table)
        console.print(sem_panel)
        console.print(qual_panel)
        console.print(segments_panel)
    
    def extract_numeric_scores(self, responses):
        """
        Extracts numeric scores from validated responses.
        Assumes each response is a string containing lines formatted as:
        'Question <number>: <score> - Explanation'
        Returns a list of lists of numeric scores.
        """
        import re
        all_scores = []
        for response in responses:
            scores = []
            lines = response.splitlines()
            for line in lines:
                match = re.search(r"Question\s+\d+:\s*(\d+)", line)
                if match:
                    try:
                        score = float(match.group(1))
                    except Exception:
                        score = 0.0
                    scores.append(score)
            if scores:
                all_scores.append(scores)
        return all_scores


# ------------------- Main Function -------------------
def main():
    """
    Production-ready main function for evaluating manga visuals.
    All intermediate results are saved to text files for auditability.
    """
    framework = LLM_MangaScoringFramework(
        guidelines_prompt="Generate guidelines to evaluate manga visuals considering art quality, character design, narrative impact, emotional tone, and cultural authenticity.",
        scale=10
    )

    questionnaire = framework.generate_questionnaire(
        num_questions=5,
        user_prompt="Focus on cultural context, narrative quality, emotional depth, and ad clarity.",
        expert_context="Evaluation should include aesthetics, cultural resonance, narrative effectiveness, and engagement."
    )
    with open("results/questionnaire.txt", "w") as f:
        f.write(questionnaire)

    image_path_A = "img1.jpg"
    image_path_B = "img1.jpg"

    responses_A, personalities_A = framework.simulate_audience_responses(
        image_path_A,
        questionnaire,
        num_profiles=5,
        audience_personality_prompt="Generate detailed personality profiles with cultural background, psychological traits, and emotional state."
    )
    with open("results/responses_A.txt", "w") as f:
        f.write("\n".join(responses_A))

    validated_responses_A = framework.enforce_guidelines(responses_A)
    with open("results/validated_responses_A.txt", "w") as f:
        f.write("\n".join(validated_responses_A))

    aggregated_score_A = framework.aggregate_scores(validated_responses_A)
    with open("results/aggregated_score_A.txt", "w") as f:
        f.write(aggregated_score_A)

    analytics_results_A = framework.advanced_analytics_pipeline(
        validated_responses_A,
        aggregated_score_A
    )
    with open("results/analytics_results_A.txt", "w") as f:
        f.write(str(analytics_results_A))

    qualitative_summary_A = framework.analyze_qualitative_feedback(validated_responses_A)
    with open("results/qualitative_summary_A.txt", "w") as f:
        f.write(qualitative_summary_A)

    numeric_scores_A = framework.extract_numeric_scores(validated_responses_A)
    reliability_A = framework.compute_inter_rater_reliability(numeric_scores_A)
    with open("results/reliability_A.txt", "w") as f:
        f.write(str(reliability_A))

    segments_A = framework.segment_responses_by_demographic(personalities_A)
    with open("results/segments_A.txt", "w") as f:
        f.write(str(segments_A))

    explanation_A = framework.explain_process(validated_responses_A, aggregated_score_A)
    with open("results/explanation_A.txt", "w") as f:
        f.write(explanation_A)

    responses_B, personalities_B = framework.simulate_audience_responses(
        image_path_B,
        questionnaire,
        num_profiles=5,
        audience_personality_prompt="Generate detailed personality profiles with cultural background, psychological traits, and emotional state."
    )
    with open("results/responses_B.txt", "w") as f:
        f.write("\n".join(responses_B))

    validated_responses_B = framework.enforce_guidelines(responses_B)
    with open("results/validated_responses_B.txt", "w") as f:
        f.write("\n".join(validated_responses_B))

    aggregated_score_B = framework.aggregate_scores(validated_responses_B)
    with open("results/aggregated_score_B.txt", "w") as f:
        f.write(aggregated_score_B)

    analytics_results_B = framework.advanced_analytics_pipeline(
        validated_responses_B,
        aggregated_score_B
    )
    with open("results/analytics_results_B.txt", "w") as f:
        f.write(str(analytics_results_B))

    qualitative_summary_B = framework.analyze_qualitative_feedback(validated_responses_B)
    with open("results/qualitative_summary_B.txt", "w") as f:
        f.write(qualitative_summary_B)

    numeric_scores_B = framework.extract_numeric_scores(validated_responses_B)
    reliability_B = framework.compute_inter_rater_reliability(numeric_scores_B)
    with open("results/reliability_B.txt", "w") as f:
        f.write(str(reliability_B))

    segments_B = framework.segment_responses_by_demographic(personalities_B)
    with open("results/segments_B.txt", "w") as f:
        f.write(str(segments_B))

    explanation_B = framework.explain_process(validated_responses_B, aggregated_score_B)
    with open("results/explanation_B.txt", "w") as f:
        f.write(explanation_B)

    p_value = framework.perform_statistical_test(
        [np.mean(scores) for scores in numeric_scores_A],
        [np.mean(scores) for scores in numeric_scores_B]
    )
    with open("results/statistical_test_p_value.txt", "w") as f:
        f.write(str(p_value))

    console = Console()
    console.print("----- Variant A Analysis -----")
    framework.display_dashboard(analytics_results_A, qualitative_summary_A, reliability_A, p_value, segments_A)

    console.print("\n----- Variant B Analysis -----")
    framework.display_dashboard(analytics_results_B, qualitative_summary_B, reliability_B, p_value, segments_B)

    console.print("\nEvaluation explanations saved in results folder.")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
from llm_service.llm_generator import generate_llm_response, generate_image_description
import logging
import numpy as np
import random
from scipy.stats import ttest_ind

# Set up basic logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Advanced Multi-Agent Manga Scoring Framework with Cognitive & Social Enhancements ---
class LLM_MangaScoringFramework:
    def __init__(self, guidelines=None, guidelines_prompt=None, scale=10, provider="openai", model="gpt-4o", temperature=0.7):
        """
        Initializes the scoring framework.
        """
        self.scale = scale
        self.provider = provider
        self.model = model
        self.temperature = temperature

        # Agent memory: dictionary mapping agent ID to a memory log.
        self.agent_memory = {}
        # Social network: dictionary mapping agent ID to list of connected agent IDs.
        self.social_network = {}  # For simplicity, we'll fill this later or randomly assign connections.

        if guidelines:
            self.guidelines = guidelines
            self.guidelines_generated = False
        elif guidelines_prompt:
            try:
                self.guidelines = generate_llm_response(guidelines_prompt, provider=self.provider, model=self.model, temperature=self.temperature)
                self.guidelines_generated = True
            except Exception as e:
                logger.error("Failed to generate guidelines: %s", e)
                self.guidelines = f"Score art quality, character design, narrative conveyance, emotional impact, and cultural authenticity on a scale of 1 to {scale} based on clarity, creativity, consistency, and cultural resonance."
                self.guidelines_generated = False
        else:
            self.guidelines = f"Score art quality, character design, narrative conveyance, emotional impact, and cultural authenticity on a scale of 1 to {scale} based on clarity, creativity, consistency, and cultural resonance."
            self.guidelines_generated = False

    # --------------------------------------
    # Cognitive Enhancements & Social Methods
    # --------------------------------------
    def simulate_agent_emotional_state(self, agent_id):
        """
        Simulates an emotional state for an agent.
        """
        emotions = ["happy", "neutral", "sad", "excited", "anxious"]
        state = random.choice(emotions)
        return state

    def update_agent_memory(self, agent_id, feedback):
        """
        Updates an agent's memory log with new feedback.
        """
        if agent_id in self.agent_memory:
            self.agent_memory[agent_id].append(feedback)
        else:
            self.agent_memory[agent_id] = [feedback]

    def simulate_social_interactions(self, agent_ids, responses):
        """
        Simulates social influence among agents.
        Adjusts responses based on connections in the social network.
        For demonstration, each agent's response is appended with a note 
        about the influence factor computed from the number of neighbors.
        """
        adjusted_responses = []
        for i, response in enumerate(responses, start=1):
            neighbors = self.social_network.get(i, [])
            influence_factor = 0.1 * len(neighbors)
            adjusted_response = response + f"\n[Social Influence Adjustment Factor: {influence_factor:.2f}]"
            adjusted_responses.append(adjusted_response)
        return adjusted_responses

    # --------------------------------------
    # Existing Methods (with Enhancements)
    # --------------------------------------
    def generate_questionnaire(self, num_questions=5, user_prompt=None, expert_context=None, questionnaire_manual=None):
        """
        Generates or uses a provided questionnaire to evaluate manga visuals.
        """
        if questionnaire_manual:
            self.questionnaire_generated = False
            return questionnaire_manual

        user_prompt = user_prompt if user_prompt else ""
        expert_context = expert_context if expert_context else ""
        # Adding environmental context (e.g., time-of-day or trending cultural events)
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
        Generates a detailed personality for an audience member using LLM.
        Now includes aspects of memory and current emotional state.
        """
        # Get current emotional state for the agent.
        current_emotion = self.simulate_agent_emotional_state(audience_index)
        # Enhance the prompt with memory and emotional state.
        enhanced_prompt = (
            f"Using insights from personality psychology (e.g., the Big Five) and cultural theories (e.g., Hofstede's dimensions), "
            f"generate a detailed personality profile for audience member {audience_index}. Include psychological traits (openness, conscientiousness, extraversion, "
            f"agreeableness, neuroticism), cultural background, and influences. Also, incorporate the agent's current emotional state: '{current_emotion}', and consider any previous feedback stored in memory. "
            f"Use the following prompt for further guidance: {personality_prompt}"
        )
        try:
            personality = generate_llm_response(enhanced_prompt, provider=self.provider, model=self.model, temperature=self.temperature)
        except Exception as e:
            logger.error("Failed to generate audience personality for member %d: %s", audience_index, e)
            personality = (
                f"Audience member {audience_index} with a balanced psychological profile, diverse cultural background, and currently feeling {current_emotion}."
            )
        return personality

    def simulate_audience_responses(self, image_path, questionnaire, num_profiles=5, audience_personalities=None, audience_personality_prompt=None):
        """
        Simulates multiple audience members, incorporating emotional state, memory, and social influence.
        """
        responses = []
        generated_personalities = []
        # For simplicity, create a dummy social network linking agents randomly.
        for i in range(1, num_profiles + 1):
            # Dummy: assign each agent 1-2 random connections (if not already assigned)
            if i not in self.social_network:
                possible_connections = list(range(1, num_profiles + 1))
                possible_connections.remove(i)
                self.social_network[i] = random.sample(possible_connections, k=min(2, len(possible_connections)))
            
            # Enhance description prompt with context.
            description_prompt = (
                "Please provide a detailed description of this manga visual. "
                "Describe the scene, characters, action, emotions, and any cultural elements depicted. "
                "Also, take into account the current cultural context and trends. "
                "Consider the following evaluation questions when describing the image:\n"
                f"{questionnaire}"
            )
            try:
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
            
            if audience_personalities is not None and len(audience_personalities) >= i:
                personality = audience_personalities[i - 1]
            elif audience_personality_prompt is not None:
                personality = self.generate_audience_personality(i, audience_personality_prompt)
            else:
                personality = f"audience member {i} with a balanced psychological profile, diverse cultural background, and insight on cultural narratives"
            generated_personalities.append(personality)
            
            # Construct evaluation prompt that now also asks for qualitative feedback.
            eval_prompt = (
                f"Based on the following tailored manga visual description:\n'{tailored_description}'\n\n"
                f"and the following questionnaire:\n{questionnaire}\n\n"
                f"Assume the role of {personality} and answer each question using a 1 to {self.scale} scale "
                f"(where 1 is very poor and {self.scale} is excellent). For each question, provide a brief justification for your score. "
                f"Also, include qualitative feedback on emotional tone, cultural relevance, and any personal reflections influenced by previous experiences.\n"
                f"Format your answer as a numbered list, with each line as: 'Question <number>: Score - Explanation'."
            )
            try:
                response = generate_llm_response(eval_prompt, provider=self.provider, model=self.model, temperature=self.temperature)
            except Exception as e:
                logger.error("Failed to generate response for audience member %d: %s", i, e)
                response = f"Response unavailable for audience member {i}."
            responses.append(response)
            # Simulate updating the agent's memory with the new response (or its key insights).
            self.update_agent_memory(i, response)
        # Simulate social influence adjustments.
        responses = self.simulate_social_interactions(list(range(1, num_profiles + 1)), responses)
        return responses, generated_personalities

    def enforce_guidelines(self, responses):
        """
        Validates that each audience response adheres to the guidelines.
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
        Aggregates scores from validated responses.
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
        Simulate extraction of scores for each evaluation criterion including cultural impact.
        """
        # For demonstration, return dummy scores.
        return {
            "art_quality": 7.0,
            "character_design": 8.0,
            "narrative_impact": 6.5,
            "emotional_impact": 7.5,
            "cultural_impact": 7.0  # New cultural criterion
        }

    def apply_mcda_ahp(self, scores, criteria_weights):
        """
        Computes a weighted aggregated score using an AHP-like approach.
        """
        total_weight = sum(criteria_weights.values())
        weighted_score = sum(scores[criterion] * criteria_weights.get(criterion, 0) for criterion in scores) / total_weight
        return weighted_score

    def apply_bayesian_updating(self, aggregated_score, prior_score=None, confidence=0.5):
        """
        Applies Bayesian updating to blend a new aggregated score with a prior score.
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
        Simulate Structural Equation Modeling (SEM) to validate latent constructs,
        including cultural authenticity as a latent variable.
        """
        sem_results = {
            "latent_constructs": {
                "visual_quality": 0.85,
                "narrative_strength": 0.78,
                "emotional_impact": 0.80,
                "cultural_authenticity": 0.75  # New latent construct for cultural aspects
            },
            "model_fit": "Good"
        }
        return sem_results

    def advanced_analytics_pipeline(self, responses, aggregated_score, prior_score=None, criteria_weights=None):
        """
        Integrates Bayesian updating, SEM validation, and MCDA (via AHP) including cultural criteria.
        """
        try:
            agg_score = float(aggregated_score)
        except (ValueError, TypeError):
            agg_score = 0.0

        # Extract criteria scores including cultural impact.
        criteria_scores = self.extract_criteria_scores(responses)
        
        # Set default criteria weights if not provided (including cultural impact).
        if criteria_weights is None:
            criteria_weights = {
                "art_quality": 0.25,
                "character_design": 0.2,
                "narrative_impact": 0.2,
                "emotional_impact": 0.2,
                "cultural_impact": 0.15
            }
        # Apply MCDA/AHP.
        mcda_score = self.apply_mcda_ahp(criteria_scores, criteria_weights)
        
        # Apply Bayesian updating.
        updated_score = self.apply_bayesian_updating(agg_score, prior_score, confidence=0.6)
        
        # Validate latent constructs using SEM.
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
        Analyzes qualitative feedback from responses (e.g., sentiment, themes, tone).
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
          responses: A list of lists or 2D numpy array of shape (n_raters, n_items)
                     where each inner list contains the numeric scores provided by a rater.
                     
        Returns:
          Cronbach's alpha as a float rounded to two decimals.
        """
        # Convert responses to a numpy array.
        data = np.array(responses)
        if data.ndim != 2:
            raise ValueError("Responses must be a 2D array (or list of lists) of numeric scores.")
        
        n_items = data.shape[1]
        # Compute variance for each item (using ddof=1 for sample variance).
        item_variances = data.var(axis=0, ddof=1)
        # Compute total scores for each rater.
        total_scores = data.sum(axis=1)
        total_variance = total_scores.var(ddof=1)
        
        # Prevent division by zero.
        if total_variance == 0:
            return 0.0
        
        # Cronbach's alpha formula.
        alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
        return round(alpha, 2)

    def perform_statistical_test(self, scores_A, scores_B):
        """
        Performs a two-sample t-test comparing two groups of aggregated scores.
        
        Parameters:
          scores_A: A list or array of aggregated scores for variant A.
          scores_B: A list or array of aggregated scores for variant B.
        
        Returns:
          p-value from the t-test as a float rounded to three decimals.
        """
        # Convert input lists to numpy arrays.
        scores_A = np.array(scores_A)
        scores_B = np.array(scores_B)
        
        # Perform an independent t-test (assuming unequal variances).
        t_stat, p_value = ttest_ind(scores_A, scores_B, equal_var=False)
        return round(p_value, 3)

    def segment_responses_by_demographic(self, personalities):
        """
        Simulates segmentation of responses by demographic/psychographic traits extracted from personality profiles.
        """
        segments = {"Segment A": [], "Segment B": []}
        for personality in personalities:
            # Dummy segmentation: based on the presence of certain keywords or randomly.
            segment = "Segment A" if "extrovert" in personality.lower() or random.random() > 0.5 else "Segment B"
            segments[segment].append(personality)
        return segments

    def display_dashboard(self, analytics_results, qualitative_summary, reliability, p_value, segments):
        """
        Displays a summary dashboard of key analytics.
        """
        dashboard = (
            "\n--- Analytics Dashboard ---\n"
            f"Initial Aggregated Score: {analytics_results.get('initial_aggregated_score')}\n"
            f"MCDA Weighted Score: {analytics_results.get('mcda_weighted_score')}\n"
            f"Bayesian Updated Score: {analytics_results.get('bayesian_updated_score')}\n"
            f"SEM Validation Results: {analytics_results.get('sem_validation')}\n"
            f"Qualitative Feedback Summary: {qualitative_summary}\n"
            f"Inter-Rater Reliability (Cronbach's alpha): {reliability}\n"
            f"Statistical Test p-value (A/B Comparison): {p_value}\n"
            f"Audience Segments: {segments}\n"
            "-----------------------------\n"
        )
        print(dashboard)

# ------------------- Main Function -------------------
def main():
    # Instantiate the scoring framework with a guidelines prompt.
    framework = LLM_MangaScoringFramework(
        guidelines_prompt="Generate guidelines to evaluate manga visuals considering art quality, character design, narrative impact, emotional tone, and cultural authenticity.",
        scale=10
    )
    
    # Generate a questionnaire with cultural, narrative, and emotional focus.
    questionnaire = framework.generate_questionnaire(
        num_questions=5,
        user_prompt="Focus on cultural context, narrative quality, emotional depth, and ad clarity.",
        expert_context="Expert evaluation should include aesthetics, cultural resonance, narrative effectiveness, and ad engagement."
    )
    
    # Define dummy image paths for two ad variants (A and B).
    image_path_A = "ad_variant_A.jpg"
    image_path_B = "ad_variant_B.jpg"
    
    # Simulate audience responses for variant A.
    responses_A, personalities_A = framework.simulate_audience_responses(
        image_path_A,
        questionnaire,
        num_profiles=5,
        audience_personality_prompt="Generate detailed personality profiles that include cultural background, psychological traits, emotional state, and memory of previous feedback."
    )
    validated_responses_A = framework.enforce_guidelines(responses_A)
    aggregated_score_A = framework.aggregate_scores(validated_responses_A)
    
    # Simulate audience responses for variant B.
    responses_B, personalities_B = framework.simulate_audience_responses(
        image_path_B,
        questionnaire,
        num_profiles=5,
        audience_personality_prompt="Generate detailed personality profiles that include cultural background, psychological traits, emotional state, and memory of previous feedback."
    )
    validated_responses_B = framework.enforce_guidelines(responses_B)
    aggregated_score_B = framework.aggregate_scores(validated_responses_B)
    
    # Run advanced analytics on variant A (using a dummy prior score).
    analytics_results_A = framework.advanced_analytics_pipeline(
        validated_responses_A,
        aggregated_score_A,
        prior_score=80.0
    )
    
    # Run advanced analytics on variant B.
    analytics_results_B = framework.advanced_analytics_pipeline(
        validated_responses_B,
        aggregated_score_B,
        prior_score=80.0
    )
    
    # Analyze qualitative feedback.
    qualitative_summary_A = framework.analyze_qualitative_feedback(validated_responses_A)
    qualitative_summary_B = framework.analyze_qualitative_feedback(validated_responses_B)
    
    # Compute inter-rater reliability.
    # For demonstration, suppose we have numeric scores from 5 raters across 5 questions for each variant.
    # In a real setting, these would be parsed from responses.
    dummy_numeric_scores_A = np.random.randint(1, 11, (5, 5)).tolist()
    dummy_numeric_scores_B = np.random.randint(1, 11, (5, 5)).tolist()
    reliability_A = framework.compute_inter_rater_reliability(dummy_numeric_scores_A)
    reliability_B = framework.compute_inter_rater_reliability(dummy_numeric_scores_B)
    
    # Perform a statistical test (e.g., t-test) between the two variants.
    # Here, we simulate multiple aggregated scores per variant.
    simulated_scores_A = [float(aggregated_score_A) + random.uniform(-2,2) for _ in range(10)]
    simulated_scores_B = [float(aggregated_score_B) + random.uniform(-2,2) for _ in range(10)]
    p_value = framework.perform_statistical_test(simulated_scores_A, simulated_scores_B)
    
    # Segment audiences by demographic/psychographic factors.
    segments_A = framework.segment_responses_by_demographic(personalities_A)
    segments_B = framework.segment_responses_by_demographic(personalities_B)
    
    # Generate an explanation of the process for variant A.
    explanation_A = framework.explain_process(validated_responses_A, aggregated_score_A)
    
    # Display dashboards for both variants.
    print("----- Variant A Analysis -----")
    framework.display_dashboard(analytics_results_A, qualitative_summary_A, reliability_A, p_value, segments_A)
    print("Explanation for Variant A:")
    print(explanation_A)
    
    print("\n----- Variant B Analysis -----")
    framework.display_dashboard(analytics_results_B, qualitative_summary_B, reliability_B, p_value, segments_B)
    print("Explanation for Variant B:")
    explanation_B = framework.explain_process(validated_responses_B, aggregated_score_B)
    print(explanation_B)

if __name__ == "__main__":
    main()

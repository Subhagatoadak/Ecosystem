from llm_service.llm_generator import generate_llm_response, generate_image_description
# Define our multi-agent scoring framework using LLM calls
# --- Multi-Agent Manga Scoring Framework ---
class LLM_MangaScoringFramework:
    def __init__(self, guidelines=None, guidelines_prompt=None, scale=10, provider="openai", model="gpt-4o", temperature=0.7):
        """
        Initializes the scoring framework.
        
        :param guidelines: (Optional) A string containing manual guidelines. If provided, these guidelines will be used.
        :param guidelines_prompt: (Optional) A prompt to generate guidelines via LLM if manual guidelines are not provided.
        :param scale: The scoring scale (default 10).
        :param provider: LLM provider to be used.
        :param model: LLM model to be used.
        :param temperature: Sampling temperature.
        """
        self.scale = scale
        self.provider = provider
        self.model = model
        self.temperature = temperature

        # If guidelines are not provided manually, generate them using the guidelines_prompt.
        if guidelines:
            self.guidelines = guidelines
            self.guidelines_generated = False
        elif guidelines_prompt:
            self.guidelines = generate_llm_response(guidelines_prompt, provider=self.provider, model=self.model, temperature=self.temperature)
            self.guidelines_generated = True
        else:
            # Default guidelines if nothing is provided.
            self.guidelines = f"Score art quality, character design, narrative conveyance, and emotional impact on a scale of 1 to {scale} based on clarity, creativity, and consistency."
            self.guidelines_generated = False

    def generate_questionnaire(self,num_questions=5,user_prompt=None, expert_context=None, questionnaire_manual=None):
        """
        Generates or uses a provided questionnaire to evaluate manga visuals.
        
        :param user_prompt: A prompt to guide the questionnaire generation.
        :param expert_context: Expert context details to be included.
        :param questionnaire_manual: (Optional) If provided, this questionnaire is used instead of generating one.
        :return: A questionnaire (string) as a numbered list.
        """
        
        if questionnaire_manual:
            self.questionnaire_generated = False
            return questionnaire_manual
        prompt = (
            f"Strictly Generate {num_questions} detailed questions only to evaluate manga visuals. Focus on aspects such as art quality, character design,\n\n "
            f"narrative impact, and emotional tone. Expert context: {expert_context}. User prompt: {user_prompt}.\n\n"
            f"Return the questions as a numbered list."
        )
        print(prompt)
        questionnaire = generate_llm_response(prompt, provider=self.provider, model=self.model, temperature=self.temperature)
        self.questionnaire_generated = True
        return questionnaire

    def generate_audience_personality(self, audience_index, personality_prompt):
        """
        Generates a detailed personality for an audience member using LLM.
        
        :param audience_index: The audience number (each represents a representative member of a different personality type).
        :param personality_prompt: A prompt to generate the personality details.
        :return: A string describing the personality.
        """
        prompt = f"Generate a detailed personality for audience member {audience_index} based on the following prompt: {personality_prompt}"
        personality = generate_llm_response(prompt, provider=self.provider, model=self.model, temperature=self.temperature)
        return personality

    def simulate_audience_responses(self, image_path, questionnaire, num_profiles=5, audience_personalities=None, audience_personality_prompt=None):
        """
        Simulates multiple audience members. For each simulated audience:
          - The image description is generated by passing a combined prompt (including the questionnaire)
            to the image description method.
          - The audience personality is either provided manually or generated via LLM.
          - Each audience member then answers the questionnaire based on the tailored description.
          
        :param image_path: Path to the manga image.
        :param questionnaire: The evaluation questionnaire (as a string).
        :param num_profiles: Number of representative audience members (each representing a different personality type).
        :param audience_personalities: (Optional) A list of manual personality details. If provided, its length should be at least num_profiles.
        :param audience_personality_prompt: (Optional) A prompt to generate personality details via LLM if manual details are not provided.
        :return: A tuple (responses, generated_personalities) where responses is a list of simulated audience responses,
                 and generated_personalities is a list of generated personality details (if applicable).
        """
        responses = []
        generated_personalities = []  # To store generated audience personalities (if applicable)
        for i in range(1, num_profiles + 1):
            # Generate tailored image description including the questionnaire context.
            description_prompt = (
                "Please provide a detailed description of this manga visual. "
                "Describe the scene, characters, action, and emotions depicted. "
                "Consider the following evaluation questions when describing the image:\n"
                f"{questionnaire}"
            )
            tailored_description = generate_image_description(
                image_path,
                description_prompt,
                provider=self.provider,
                model="gpt-4o-mini",
                temperature=self.temperature
            )
            # Determine audience personality.
            # Audience number i represents a representative member of a distinct personality type.
            if audience_personalities is not None and len(audience_personalities) >= i:
                personality = audience_personalities[i-1]
            elif audience_personality_prompt is not None:
                personality = self.generate_audience_personality(i, audience_personality_prompt)
            else:
                personality = f"a diverse audience member with profile {i} who has a unique perspective on manga art"
            generated_personalities.append(personality)
            # Create prompt for the audience to answer the questionnaire based on the tailored description.
            prompt = (
                f"Based on the following tailored manga visual description:\n'{tailored_description}'\n\n"
                f"and the following questionnaire:\n{questionnaire}\n\n"
                f"Assume the role of {personality} and answer each question using a 1 to {self.scale} scale "
                f"(where 1 is very poor and {self.scale} is excellent). For each question, provide a brief justification for your score.\n"
                f"Format your answer as a numbered list, with each line as: 'Question <number>: Score - Explanation'."
            )
            response = generate_llm_response(prompt, provider=self.provider, model=self.model, temperature=self.temperature)
            responses.append(response)
        return responses, generated_personalities

    def enforce_guidelines(self, responses):
        """
        Checks that each simulated audience response adheres to the scoring guidelines.
        """
        prompt = (
            f"Review the following simulated audience responses:\n{responses}\n\n"
            f"Ensure that each score is between 1 and {self.scale} and that the justifications align with the following guidelines: {self.guidelines}.\n"
            f"If any score is out of range or the justification does not follow the guidelines, provide corrections. Otherwise, confirm that the responses are valid.\n"
            f"Return the validated responses."
        )
        validated_responses = generate_llm_response(prompt, provider=self.provider, model=self.model, temperature=self.temperature)
        return validated_responses

    def aggregate_scores(self, responses):
        """
        Extracts numeric scores from the validated responses and computes the average score.
        """
        prompt = (
            f"Given the following validated audience responses (each containing scores in the format 'Question <number>: Score - Explanation'):\n{responses}\n\n"
            f"Extract all the numeric scores and compute the average score for the manga visual. Return only the final aggregated score. The final score is always normalized to be out of 100."
        )
        final_score = generate_llm_response(prompt, provider=self.provider, model=self.model, temperature=self.temperature)
        return final_score

    def explain_process(self, responses, final_score):
        """
        Provides a detailed explanation of how the final score was derived.
        """
        prompt = (
            f"Explain in detail how the final aggregated score of {final_score} was derived from the following audience responses:\n{responses}\n\n"
            f"Include how each question and its corresponding score contributed to the final score, any corrections made during guideline enforcement, "
            f"and highlight key insights from the process, any areas of improvement based on audience review of questionnaire,anything that has worked, anything that is completely off. Provide the explanation for the final score."
        )
        explanation = generate_llm_response(prompt, provider=self.provider, model=self.model, temperature=self.temperature)
        return explanation

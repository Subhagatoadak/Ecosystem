import sys
from AgentsFramework.VisualScoringAgent.VisualScoringAgent import *
# Main execution block
def main():
    
    image_path = "img1.jpg"
    
    # Flag to determine whether to display generated components (if generated by LLM)
    show_generated_components = True
    
    # Parameters that can be manually provided or generated via LLM:
    # 1. Questionnaire: either manual or generated using user_prompt & expert_context.
    questionnaire_manual = None  # Set to a manual questionnaire string if desired.
    user_prompt = "Focus on dynamic action scenes and vibrant color usage."
    expert_context = "The visuals should clearly convey movement and emotion with well-composed action sequences."
    
    # 2. Guidelines: either manual or generated using a guidelines prompt.
    guidelines_manual = None  # Set to a manual guidelines string if desired.
    guidelines_prompt = "Generate guidelines for scoring manga visuals based on art quality, character design, narrative, and emotional impact."
    
    # 3. Audience Personalities: either a list of manual personality details or a personality prompt.
    audience_personalities_manual = None  # If provided, this should be a list of personality descriptions.
    audience_personality_prompt = "Describe a unique personality that is passionate about manga art and has strong opinions on visual storytelling."
    
    # Initialize the scoring framework.
    framework = LLM_MangaScoringFramework(
        guidelines=guidelines_manual,
        guidelines_prompt=guidelines_prompt,
        scale=10,           
        provider="openai",
        model="gpt-4o",
        temperature=0.7
    )
    
    # Display generated guidelines if requested.
    if show_generated_components and framework.guidelines_generated:
        print("=== Generated Guidelines ===")
        print(framework.guidelines, "\n")
    
    # Generate or use manual questionnaire.
    questionnaire = framework.generate_questionnaire(num_questions=5,user_prompt=user_prompt, expert_context=expert_context, questionnaire_manual=questionnaire_manual)
    if show_generated_components and framework.questionnaire_generated:
        print("=== Generated Questionnaire ===")
        print(questionnaire, "\n")
    
    # Simulate audience responses.
    # audience_personalities_manual is optional; if provided, its length should be at least the number of profiles.
    responses, generated_personalities = framework.simulate_audience_responses(
        image_path,
        questionnaire,
        num_profiles=5,
        audience_personalities=audience_personalities_manual,
        audience_personality_prompt=audience_personality_prompt
    )
    # If personalities were generated by LLM, display them.
    if show_generated_components and (audience_personalities_manual is None):
        print("=== Generated Audience Personalities ===")
        for idx, personality in enumerate(generated_personalities, 1):
            print(f"Audience Member {idx}: {personality}")
        print()
    
    print("=== Simulated Audience Responses ===")
    for idx, resp in enumerate(responses, 1):
        print(f"Response {idx}:\n{resp}\n")
    
    # Enforce guidelines on the responses.
    validated_responses = framework.enforce_guidelines(responses)
    print("=== Validated Audience Responses ===")
    print(validated_responses, "\n")
    
    # Aggregate the scores.
    final_score = framework.aggregate_scores(validated_responses)
    print("=== Final Aggregated Score ===")
    print(final_score, "\n")
    
    # Explain the scoring process.
    explanation = framework.explain_process(validated_responses, final_score)
    print("=== Explanation of the Scoring Process ===")
    print(explanation)

if __name__ == "__main__":
    main()
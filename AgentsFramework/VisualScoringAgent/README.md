
# Manga Visual Evaluation Framework

The **Manga Visual Evaluation Framework** is a multi-agent system designed to simulate a diverse audience for evaluating and scoring manga visuals. Leveraging large language models (LLMs) (e.g., OpenAI's GPT-4), the framework automatically generates detailed image descriptions, evaluation questionnaires, scoring guidelines, and personality profiles for simulated audience members. It then aggregates audience responses to produce a final score and offers a transparent explanation of the evaluation process.

---

## Expert Foundation

**This project is built on the foundational work of manga expert [Ananya Saha](https://www.drotaku.in/).**  
Ananya Saha is the driving force behind the evaluation criteria and the overall design of this framework. Her deep insights into manga art, storytelling, and visual narrative have been integral to the creation and ongoing refinement of this system.  
For a deeper understanding of this repository and its underlying principles, you can connect with her on [LinkedIn](https://www.linkedin.com/in/ananya-saha-phd-dr-otaku-6259436b/).

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Inputs and Outputs](#inputs-and-outputs)
- [Improvement Areas](#improvement-areas)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The framework is designed to analyze manga visuals by simulating real audience feedback. Its primary goals are to:

- **Dynamically Evaluate:** Generate detailed questionnaires and scoring guidelines either automatically via LLMs or through manual input.
- **Simulate a Diverse Audience:** Model varied audience personalities by either providing manual personality details or generating them using LLM prompts.
- **Process Images:** Produce detailed descriptions of manga visuals from image files by encoding and processing them through LLMs.
- **Aggregate Scores:** Collect individual scores, validate responses against guidelines, and compute a final aggregated score.
- **Ensure Transparency:** Offer a clear explanation of how the final score was derived.

---

## Features

- **Flexible Inputs:**  
  - **Guidelines:** Supply manual scoring guidelines or generate them using LLM prompts.
  - **Questionnaire:** Either provide a custom questionnaire or have the LLM generate one based on a user prompt and expert context.
  - **Audience Personalities:** Customize personality profiles manually or generate them via LLM prompts.
  - **Image Input:** Accept an image file path to generate a detailed description of the manga visual.

- **Multi-Agent Simulation:**  
  Each audience member, representing a unique personality type, receives a tailored image description (integrating the questionnaire) and evaluates the visual on a consistent 1-to-10 scale.

- **LLM-Powered Processing:**  
  The system utilizes LLMs to generate:
  - Detailed image descriptions (using OpenAI's image description method with base64 encoding).
  - Questionnaires and scoring guidelines.
  - Audience personality profiles.
  - Explanations of the scoring process.

- **Detailed Reporting:**  
  The framework aggregates scores and provides a comprehensive explanation of the evaluation process, ensuring complete transparency.

- **Modular and Extensible:**  
  Easily swap or enhance individual components (e.g., change LLM providers or adjust evaluation criteria).

---

## Architecture

The framework is composed of several key components:

1. **Image Description Module:**  
   - **Function:** Encodes an input image, sends it along with a tailored prompt to OpenAI's API, and retrieves a detailed textual description.
   - **Usage:** Acts as the basis for evaluating the manga visual.

2. **Questionnaire Generator:**  
   - **Function:** Generates evaluation questions based on a user prompt and expert context, or accepts a manually defined questionnaire.
   - **Output:** A numbered list of questions focusing on art quality, character design, narrative impact, and emotional tone.

3. **Guidelines Generator:**  
   - **Function:** Generates or accepts scoring guidelines that ensure consistency in evaluation.
   - **Output:** A guideline string used to validate audience responses.

4. **Audience Personality Module:**  
   - **Function:** Generates or accepts manual personality profiles for each simulated audience member.
   - **Note:** Each audience number represents a representative member of a distinct personality type.

5. **Audience Response Simulator:**  
   - **Function:** Combines the tailored image description with the questionnaire and personality details to simulate audience responses.
   - **Output:** Simulated responses with scores and justifications for each evaluation question.

6. **Guideline Enforcement & Aggregation Module:**  
   - **Function:** Validates that audience responses adhere to the scoring guidelines and aggregates scores to compute a final result.
   - **Output:** Final aggregated score and a detailed explanation of the scoring process.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/manga-visual-evaluation-framework.git
   cd manga-visual-evaluation-framework
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.7 or later. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` should include:
   - `openai`
   - `python-dotenv`
   - `requests`
   - (Other dependencies as required)

3. **Set Up Environment Variables:**

   Create a `.env` file in the project root directory with the following content:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key  # (if needed)
   ANTHROPIC_API_KEY=your_anthropic_api_key        # (if needed)
   GEMINI_API_KEY=your_gemini_api_key              # (if needed)
   ```

---

## Configuration

The framework supports flexible configuration:
- **Manual vs. Generated Inputs:**  
  - Supply manual guidelines, questionnaires, and audience personality details, or provide prompts for LLM generation.
- **Display Options:**  
  - Enable a flag (`show_generated_components`) to print generated guidelines, questionnaires, or audience personality profiles for review.

Edit the configuration parameters in the main execution block (`script.py`) as required.

---

## Usage

Run the framework from the command line by providing the path to a manga image:

```bash
python script.py 
```

### Workflow

1. **Image Description:**  
   The system encodes the input image and sends it along with a detailed prompt (that includes questionnaire context) to generate a comprehensive image description.

2. **Questionnaire & Guidelines:**  
   - The framework uses manually provided or LLM-generated questionnaires and guidelines.
   - Generated components are displayed if `show_generated_components` is enabled.

3. **Simulated Audience Responses:**  
   - Each representative audience member (each with a unique personality) receives a tailored image description.
   - The audience then responds to the questionnaire based on that description.
   - The system supports manual or LLM-generated personality profiles.

4. **Validation & Aggregation:**  
   - Audience responses are validated against the scoring guidelines.
   - Scores are aggregated to produce a final score.
   - A detailed explanation of the scoring process is generated.

5. **Output:**  
   The final aggregated score, validated responses, and a detailed explanation are printed to the console.

---

## Inputs and Outputs

### Inputs
- **Image File:**  
  A path to a manga image (e.g., `manga_image.jpg`).

- **Questionnaire Parameters:**  
  - **User Prompt:** e.g., "Focus on dynamic action scenes and vibrant color usage."
  - **Expert Context:** e.g., "The visuals should clearly convey movement and emotion with well-composed action sequences."
  - **(Optional) Manual Questionnaire:** A predefined questionnaire string.

- **Guidelines Parameters:**  
  - **(Optional) Manual Guidelines:** A predefined guidelines string.
  - **Guidelines Prompt:** e.g., "Generate guidelines for scoring manga visuals based on art quality, character design, narrative, and emotional impact."

- **Audience Personality Parameters:**  
  - **(Optional) Manual Audience Personalities:** A list of personality descriptions.
  - **Audience Personality Prompt:** e.g., "Describe a unique personality that is passionate about manga art and has strong opinions on visual storytelling."

### Outputs
- **Generated Image Description:**  
  A detailed description of the manga visual based on the provided image.

- **Generated/Provided Questionnaire:**  
  A numbered list of evaluation questions.

- **Generated/Provided Guidelines:**  
  A set of scoring guidelines for audience responses.

- **Simulated Audience Responses:**  
  Each audience member (representing a distinct personality type) produces scores and justifications for each evaluation question.

- **Validated Audience Responses:**  
  Audience responses that are checked against the scoring guidelines.

- **Final Aggregated Score:**  
  The average score computed from all audience responses.

- **Process Explanation:**  
  A detailed explanation of how the final score was derived, including insights and corrections made during validation.

- **(Optional) Display of Generated Components:**  
  If enabled, generated guidelines, questionnaire, and audience personality profiles are printed.

---

## Improvement Areas

While this framework is a robust starting point for evaluating manga visuals using LLM-based simulation, several areas can be enhanced:

1. **Enhanced Image Processing:**
   - Integrate dedicated image captioning models that accept image binaries directly.
   - Improve error handling during image upload and encoding.

2. **LLM Provider Flexibility:**
   - Support additional LLM providers such as Anthropic’s Claude or Google Gemini.
   - Implement a fallback mechanism if one provider fails.

3. **Response Parsing and Validation:**
   - Enhance parsing of audience responses to reliably extract numeric scores.
   - Use advanced natural language processing to validate and correct responses more effectively.

4. **User Interface:**
   - Develop a web-based or GUI interface for a more user-friendly experience.
   - Include visualizations for aggregated scores and process explanations.

5. **Customization and Extensibility:**
   - Allow more granular control over evaluation criteria.
   - Enable saving/loading of custom guidelines, questionnaires, and audience profiles for reuse.

6. **Scalability:**
   - Optimize API calls and parallelize audience simulation for larger volumes.
   - Integrate caching for repeated image descriptions or LLM responses.

7. **Security and Compliance:**
   - Ensure secure management of sensitive API keys and user data.
   - Comply with data privacy regulations when integrating with third-party services.

---

## Contributing

Contributions are welcome! If you have ideas for improvements, bug fixes, or additional features, please open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push your branch.
4. Open a pull request describing your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

By combining the power of LLMs with a modular, multi-agent design, this framework provides a flexible and transparent approach to evaluating manga visuals. Whether you choose to supply your own guidelines, questionnaires, and audience personalities or let the LLM generate them, the framework adapts to your needs and produces reproducible results.

**Happy Evaluating!**
```
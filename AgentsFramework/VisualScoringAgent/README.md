# Manga Visual Evaluation Framework

The **Manga Visual Evaluation Framework** is a multi-agent system designed to simulate a diverse audience to evaluate and score manga visuals. The framework leverages large language models (LLMs) (e.g., OpenAI's GPT-4) to generate image descriptions, evaluation questionnaires, scoring guidelines, and detailed personality profiles for simulated audience members. It then aggregates audience responses to produce a final score and provides a transparent explanation of the evaluation process.

> **Expert Contribution:**  
> Manga expert [Ananya Saha](https://www.drotaku.in/) (LinkedIn: [Ananya Saha](https://www.linkedin.com/in/ananya-saha-phd-dr-otaku-6259436b/)) is an **integral part** of this process. Her deep insights into manga art and storytelling are embedded in the evaluation criteria and have been critical to refining the framework.

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

This framework is built to analyze manga visuals by simulating audience feedback. Its primary objectives are:

- **Dynamic Evaluation:** Generate detailed questionnaires and scoring guidelines automatically using LLMs or use manual inputs.
- **Diverse Audience Simulation:** Model diverse audience personalities, either by providing manual personality descriptions or generating them via LLM prompts.
- **Image Analysis:** Generate a detailed description of the manga visual from an image file by encoding it and sending it (with additional context) to an LLM.
- **Scoring and Aggregation:** Collect individual audience scores, enforce scoring guidelines, and compute a final aggregated score.
- **Transparency:** Provide detailed explanations for how the final score was derived.

> **Expert Contribution:**  
> Manga expert [Ananya Saha](https://www.drotaku.in/) is an integral part of this process. Her expertise in manga art and narrative significantly informs the evaluation criteria, ensuring that the framework's outcomes are both rigorous and relevant.

---

## Features

- **Flexible Inputs:**  
  - **Guidelines:** Manually provide scoring guidelines or generate them using a prompt.
  - **Questionnaire:** Either supply a custom questionnaire or let the LLM generate one based on user prompt and expert context.
  - **Audience Personalities:** Customize personality details manually or generate them via LLM prompts.
  - **Image Input:** Accept an image file path to generate a description of the manga visual.

- **Multi-Agent Simulation:**  
  Each audience member (representative of a distinct personality type) receives a tailored image description (integrating the questionnaire) and evaluates the visual on a consistent 1-to-10 scale.

- **LLM-Powered Processing:**  
  The system uses LLMs to generate natural language outputs for:
  - Image descriptions (using OpenAI's image description method with base64 encoding).
  - Questionnaires.
  - Guidelines.
  - Audience personality profiles.
  - Explanations of the scoring process.

- **Detailed Reporting:**  
  The framework aggregates individual scores and provides a comprehensive explanation of the evaluation process, ensuring transparency.

- **Extensible and Modular:**  
  The design is modular, allowing you to swap or improve individual components (e.g., change the LLM provider or adjust the evaluation criteria).

---

## Architecture

The framework consists of the following key components:

1. **Image Description Module:**  
   - **Functionality:** Encodes an input image, sends it with a tailored prompt to OpenAI's API, and retrieves a detailed textual description.
   - **Usage:** Used as the basis for evaluating the manga visual.

2. **Questionnaire Generator:**  
   - **Functionality:** Generates a list of evaluation questions based on user prompt and expert context, or accepts a manual questionnaire.
   - **Output:** A numbered list of questions focusing on art quality, character design, narrative impact, and emotional tone.

3. **Guidelines Generator:**  
   - **Functionality:** Generates or accepts scoring guidelines that define the evaluation criteria and ensure consistency in scoring.
   - **Output:** A guideline string that is used to validate audience responses.

4. **Audience Personality Module:**  
   - **Functionality:** Generates or accepts manual personality profiles for each simulated audience member.  
   - **Note:** Each audience number represents a representative member of a distinct personality type.

5. **Audience Response Simulator:**  
   - **Functionality:** For each audience member, it combines the tailored image description with the questionnaire and personality details.  
   - **Output:** A simulated response with scores and justifications for each question.

6. **Guideline Enforcement & Aggregation Module:**  
   - **Functionality:** Validates that audience responses adhere to guidelines and aggregates the scores to compute a final score.
   - **Output:** A final aggregated score and a detailed explanation of the scoring process.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/manga-visual-evaluation-framework.git
   cd manga-visual-evaluation-framework
   ```

2. **Install Dependencies:**

   This project requires Python 3.7 or later. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` should include:
   - `openai`
   - `python-dotenv`
   - `requests`
   - (Any other dependencies required by your project)

3. **Environment Variables:**

   Create a `.env` file in the project root directory with the following variables:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key  # (if needed)
   ANTHROPIC_API_KEY=your_anthropic_api_key        # (if needed)
   GEMINI_API_KEY=your_gemini_api_key              # (if needed)
   ```

---

## Configuration

The framework allows flexible configuration:
- **Manual vs. Generated Inputs:**  
  - You can supply manual inputs for guidelines, questionnaires, and audience personalities.
  - Alternatively, you can provide prompts to generate these components using the LLM.
- **Display Options:**  
  - Set a flag (`show_generated_components`) to `True` to print out any generated guidelines, questionnaires, or audience personality profiles for review.

Edit the parameters in the main execution block of `script.py` as needed.

---

## Usage

Run the framework from the command line by providing the path to a manga image:

```bash
python script.py path/to/manga_image.jpg
```

### Workflow

1. **Image Description:**  
   The system encodes the input image and sends it along with a detailed prompt (including questionnaire context) to generate a description.

2. **Questionnaire and Guidelines:**  
   - The framework either uses manually provided questionnaires and guidelines or generates them via LLM prompts.
   - If generated, these components are displayed (if `show_generated_components` is enabled).

3. **Simulated Audience Responses:**  
   - For each representative audience member (each with a unique personality), a tailored image description is generated.
   - The audience then answers the questionnaire based on that description.
   - The user can either provide manual personality details or let the LLM generate them.

4. **Validation and Aggregation:**  
   - Audience responses are validated against the provided/generated guidelines.
   - Scores are aggregated to produce a final score.
   - A detailed explanation of the scoring process is generated.

5. **Output:**  
   The final aggregated score, validated responses, and an explanation of the process are printed to the console.

---

## Inputs and Outputs

### Inputs
- **Image File:**  
  A path to a manga image (e.g., `manga_image.jpg`).

- **Questionnaire Parameters:**  
  - **User Prompt:** e.g., "Focus on dynamic action scenes and vibrant color usage."
  - **Expert Context:** e.g., "The visuals should clearly convey movement and emotion with well-composed action sequences."
  - **(Optional) Manual Questionnaire:** A pre-defined questionnaire string.

- **Guidelines Parameters:**  
  - **(Optional) Manual Guidelines:** A pre-defined guidelines string.
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
  Audience responses that have been checked against the scoring guidelines.

- **Final Aggregated Score:**  
  The average score computed from all audience responses.

- **Process Explanation:**  
  A detailed explanation of how the final score was derived, including insights and corrections made during the validation process.

- **(Optional) Generated Components Display:**  
  If enabled, the generated guidelines, questionnaire, and audience personality profiles are printed.

---

## Improvement Areas

While the current framework provides a robust starting point for evaluating manga visuals using LLM-based simulation, there are several areas for improvement:

1. **Enhanced Image Processing:**
   - Integrate a dedicated image captioning model or service that accepts image binaries directly.
   - Improve error handling for image upload/encoding.

2. **LLM Provider Flexibility:**
   - Add support for additional LLM providers such as Anthropic’s Claude or Google Gemini.
   - Provide a fallback mechanism if one provider fails.

3. **Response Parsing and Validation:**
   - Improve the parsing of audience responses to extract numeric scores more reliably.
   - Implement advanced natural language processing techniques to better validate and correct responses.

4. **User Interface:**
   - Develop a web-based or GUI interface for ease of use.
   - Provide visualizations for aggregated scores and process explanations.

5. **Customization and Extensibility:**
   - Allow more granular control over the evaluation criteria.
   - Enable saving and loading of custom guidelines, questionnaires, and audience profiles for repeated use.

6. **Scalability:**
   - Optimize API calls and parallelize audience simulation to handle larger volumes.
   - Integrate caching mechanisms for repeated image descriptions or LLM responses.

7. **Security and Compliance:**
   - Ensure that sensitive API keys and user data are securely managed.
   - Comply with data privacy regulations when integrating with third-party services.

---

## Contributing

Contributions are welcome! If you have ideas for improvement, bug fixes, or additional features, please open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push your branch.
4. Open a pull request describing your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

By combining the power of LLMs with a modular, multi-agent design, this framework provides a unique and flexible approach to evaluating manga visuals. Whether you choose to supply your own guidelines, questionnaires, and audience personalities or let the LLM generate them, the framework adapts to your needs and produces transparent, reproducible results.

Happy evaluating!
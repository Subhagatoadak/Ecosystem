# SQL Query Generation Agent

A multi-agent, production-grade solution for generating complex SQL queries from natural language input. This project integrates multiple LLM providers (OpenAI, Hugging Face, Anthropic Claude, and a hypothetical Google Gemini) with a modular, multi-department architecture that extracts database schema details, determines join strategies, generates SQL queries, validates syntax, and evaluates query performance. A centralized logging system and a supervisor component oversee inter-agent communication for enhanced reliability and debugging.

---

## Features

- **Multi-Provider LLM Support:**  
  Easily switch between providers like OpenAI, Hugging Face, Claude, and Gemini using configuration in the environment.

- **Modular Agent Architecture:**  
  Separate departments for schema extraction, join determination (with planning for JOINs, CTEs, subqueries, or correlated subqueries), SQL query generation, validation, and evaluation.

- **Centralized Discussion Logging & Supervision:**  
  All departments log their decisions and errors to a centralized logger. A supervisor agent reviews these logs for inconsistencies or errors.

- **Robust Error Handling:**  
  Each agent includes error handling to catch and log issues, ensuring graceful failure and easier debugging.

- **Flexible Integration:**  
  Easily integrate with any SQLite database by setting the database path via environment variables.

---

## Directory Structure

```
.
├── llm_generator.py       # Contains helper functions for interacting with various LLM providers and image encoding.
├── SQLAgent.py         # Implements the multi-agent SQL query generation system (schema extraction, join determination, query generation, validation, evaluation, logging, and supervision).
├── main_sql.py              # Entry point for the application. Connects to the database, instantiates the agents, processes user input, and outputs results.
├── README.md            # This file.
└── .env                 # Environment variable configuration (not provided, see Setup section).
```

---

## Setup & Prerequisites

### Prerequisites

- **Python 3.7+**
- Required packages:
  - `openai`
  - `requests`
  - `python-dotenv`
  - `sqlparse` (optional for formatting)
- An SQLite database file (default: `example.db`).

### Environment Variables

Create a `.env` file in the project root with the following variables:

```dotenv
# API keys for LLM providers
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here  # Hypothetical key

# SQLite database file path (optional, defaults to "example.db")
SQLITE_DB_PATH=your_database_path_here
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/sql-query-agent.git
   cd sql-query-agent
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   _Note: If a `requirements.txt` file is not provided, manually install packages:_

   ```bash
   pip install openai requests python-dotenv sqlparse
   ```

4. **Set up your `.env` file** as described in the Environment Variables section.

---

## Running the Application

Run the application using:

```bash
python main_sql.py
```

- The application will connect to the specified SQLite database.
- It prompts you for a natural language query.
- It then processes the query through schema extraction, join determination (with planning), SQL query generation, validation, and evaluation.
- Finally, it prints the generated SQL query, the query evaluation (execution plan), and the discussion log for debugging and auditing.

---

## Code Details

### `llm_generator.py`

- **Purpose:**  
  Provides helper functions to interact with multiple LLM providers (OpenAI, Hugging Face, Anthropic Claude, and Google Gemini). Also includes image encoding functions used for generating image descriptions.
- **Key Functions:**
  - `generate_llm_response(prompt, provider, model, temperature)`: Retrieves LLM-generated responses.
  - `encode_image(image_path)`: Encodes an image in base64.
  - `generate_image_description(...)`: Generates image descriptions by sending the image data along with a prompt.

### `SQLAgent.py`

- **Purpose:**  
  Implements the multi-agent SQL query generation system.
- **Key Components:**
  - **DiscussionLoggerAgent:** Centralized logging for inter-agent communication.
  - **SchemaExtractionAgent:** Extracts tables, columns, and foreign key relationships from the SQLite database.
  - **JoinDeterminationAgent:** Analyzes the user query and schema to determine whether JOINs, CTEs, subqueries, or correlated subqueries are needed. Returns a structured plan with planning and implementation details.
  - **QueryGenerationAgent:** Uses the join planning details and schema to generate an optimized SQL query.
  - **SQLValidationAgent:** Validates the generated SQL syntax via an `EXPLAIN` statement.
  - **QueryEvaluationAgent:** Retrieves the query execution plan using `EXPLAIN QUERY PLAN`.
  - **SupervisorAgent:** Reviews discussion logs for any errors.
  - **SQLQueryOrchestrator:** Coordinates the entire process from schema extraction to final query evaluation.

### `main_sql.py`

- **Purpose:**  
  The entry point of the application. It connects to the database, initializes the logger and orchestrator, accepts user input, and displays the results.
- **Usage:**  
  The user is prompted to enter a natural language query. The orchestrator processes the query and outputs the final SQL query, its evaluation, and discussion logs.

---

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues if you encounter bugs or have feature suggestions.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/my-new-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/my-new-feature`).
5. Create a new Pull Request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Additional Notes

- The Google Gemini integration is hypothetical; adjust as needed if and when the API becomes publicly available.
- Ensure that your environment variables are kept secure and not exposed publicly.
- For production deployments, consider additional error handling, logging enhancements, and secure API key management.

---

Happy querying!
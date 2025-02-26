import sqlite3
import datetime
import logging
from llm_service.llm_generator import generate_llm_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DiscussionLoggerAgent:
    """
    Logs discussions from various departments.
    """
    def __init__(self):
        self.logs = []

    def log(self, department, message, level="INFO"):
        timestamp = datetime.datetime.now().isoformat()
        log_entry = f"{timestamp} [{department}] {level}: {message}"
        self.logs.append(log_entry)
        if level == "ERROR":
            logging.error(log_entry)
        elif level == "WARNING":
            logging.warning(log_entry)
        else:
            logging.info(log_entry)

    def get_logs(self):
        return self.logs

class SchemaExtractionAgent:
    """
    Department: Schema
    Extracts the database schema including tables, columns, and foreign key relationships.
    """
    def __init__(self, connection, logger):
        self.connection = connection
        self.logger = logger
        self.department = "Schema Department"

    def extract_schema(self):
        try:
            schema = {}
            cursor = self.connection.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            if not tables:
                self.logger.log(self.department, "No tables found in the database.", "WARNING")
            for table in tables:
                cursor = self.connection.execute(f"PRAGMA table_info({table});")
                columns = [row[1] for row in cursor.fetchall()]
                fk_cursor = self.connection.execute(f"PRAGMA foreign_key_list({table});")
                foreign_keys = []
                for row in fk_cursor.fetchall():
                    foreign_keys.append({
                        "table": row[2],
                        "from": row[3],
                        "to": row[4]
                    })
                schema[table] = {
                    "columns": columns,
                    "foreign_keys": foreign_keys
                }
            self.logger.log(self.department, "Schema extracted successfully.")
            return schema
        except Exception as e:
            self.logger.log(self.department, f"Error extracting schema: {e}", "ERROR")
            raise

class JoinDeterminationAgent:
    """
    Department: Join
    Analyzes the user query and schema to decide if the query requires a JOIN,
    a CTE, a subquery, or a correlated subquery.
    Returns a structured plan with both planning (reasoning and recommendation)
    and implementation details.
    """
    def __init__(self, schema, logger):
        self.schema = schema
        self.logger = logger
        self.department = "Join Department"

    def _format_schema(self):
        lines = []
        for table, info in self.schema.items():
            lines.append(f"Table: {table}")
            lines.append("  Columns: " + ", ".join(info["columns"]))
            if info["foreign_keys"]:
                lines.append("  Foreign Keys:")
                for fk in info["foreign_keys"]:
                    lines.append(f"    {table}.{fk['from']} -> {fk['table']}.{fk['to']}")
            lines.append("")
        return "\n".join(lines)

    def determine_joins(self, user_query):
        try:
            schema_str = self._format_schema()
            prompt = f"""
You are a database schema and query planning expert. Analyze the following natural language query and database schema. Your task is to decide if the query requires:
  - Table JOINs,
  - A Common Table Expression (CTE),
  - A subquery, or
  - A correlated subquery.

Please provide a structured response with two sections:
1. Planning: Explain your reasoning and state which SQL construct you recommend (JOIN, CTE, subquery, or correlated subquery) and why.
2. Implementation Details: If JOINs are recommended, list the tables to join and the join conditions (e.g., "Tables: tableA, tableB; Conditions: tableA.colX = tableB.colY"). If a CTE or subquery is recommended, briefly describe its structure.

User Query: "{user_query}"

Database Schema:
{schema_str}

Respond in the following format:

Planning: <your reasoning and recommendation>
Implementation Details: <the join/subquery/CTE details>
"""
            result = generate_llm_response(prompt, provider="openai", model="gpt-4o", temperature=0)
            self.logger.log(self.department, f"Join planning and details determined: {result}")
            return result
        except Exception as e:
            self.logger.log(self.department, f"Error determining joins: {e}", "ERROR")
            raise

class QueryGenerationAgent:
    """
    Department: Query Generation
    Generates the final SQL query using the user query, join planning details, and schema.
    """
    def __init__(self, schema, logger):
        self.schema = schema
        self.logger = logger
        self.department = "Query Generation Department"
        self.schema_str = self._format_schema()

    def _format_schema(self):
        lines = []
        for table, info in self.schema.items():
            lines.append(f"Table: {table}")
            lines.append("  Columns: " + ", ".join(info["columns"]))
            if info["foreign_keys"]:
                lines.append("  Foreign Keys:")
                for fk in info["foreign_keys"]:
                    lines.append(f"    {table}.{fk['from']} -> {fk['table']}.{fk['to']}")
            lines.append("")
        return "\n".join(lines)

    def generate_query(self, user_query, join_info):
        try:
            prompt = f"""
You are an expert SQL query generator. Generate an optimized, syntactically correct SQL query for the following user request.
User Query: "{user_query}"
Join Planning and Details: "{join_info}"
Database Schema:
{self.schema_str}
Ensure that:
- The query uses the recommended SQL construct (JOIN, CTE, subquery, or correlated subquery) as indicated in the join planning.
- All referenced tables and columns exist in the schema.
- Appropriate SQL constructs (JOINs, CTEs, subqueries) are used.
SQL Query:
"""
            sql_query = generate_llm_response(prompt, provider="openai", model="gpt-4o", temperature=0)
            self.logger.log(self.department, f"SQL query generated: {sql_query}")
            return sql_query
        except Exception as e:
            self.logger.log(self.department, f"Error generating SQL query: {e}", "ERROR")
            raise

class SQLValidationAgent:
    """
    Department: Validation
    Validates the generated SQL query by checking its syntax using an EXPLAIN statement.
    """
    def __init__(self, connection, logger):
        self.connection = connection
        self.logger = logger
        self.department = "Validation Department"

    def validate_query(self, sql_query):
        try:
            self.connection.execute("EXPLAIN " + sql_query)
            self.logger.log(self.department, "SQL query validated successfully.")
            return True, None
        except Exception as e:
            self.logger.log(self.department, f"SQL validation error: {e}", "ERROR")
            return False, str(e)

class QueryEvaluationAgent:
    """
    Department: Evaluation
    Evaluates the SQL query by retrieving its execution plan using EXPLAIN QUERY PLAN.
    """
    def __init__(self, connection, logger):
        self.connection = connection
        self.logger = logger
        self.department = "Evaluation Department"

    def evaluate_query(self, sql_query):
        try:
            cursor = self.connection.execute("EXPLAIN QUERY PLAN " + sql_query)
            plan = cursor.fetchall()
            self.logger.log(self.department, "Query evaluated successfully.")
            return plan
        except Exception as e:
            self.logger.log(self.department, f"Error evaluating query: {e}", "ERROR")
            return f"Error evaluating query: {e}"

class SupervisorAgent:
    """
    Department: Supervisor
    Oversees and reviews the discussions (logs) from all departments.
    """
    def __init__(self, logger):
        self.logger = logger
        self.department = "Supervisor Department"

    def review_discussions(self):
        errors = [log for log in self.logger.get_logs() if "ERROR" in log]
        if errors:
            self.logger.log(self.department, f"Supervisor noticed errors: {errors}", "WARNING")
        else:
            self.logger.log(self.department, "No errors detected in discussions.", "INFO")
        return errors

class SQLQueryOrchestrator:
    """
    Coordinates all departmental agents to process a user query by:
      1. Extracting the schema.
      2. Determining join/planning information.
      3. Generating the SQL query.
      4. Validating the query.
      5. Evaluating the query.
      6. Having the supervisor review the discussions.
    """
    def __init__(self, connection, logger):
        self.connection = connection
        self.logger = logger
        self.schema_agent = SchemaExtractionAgent(connection, logger)
        self.schema = None
        self.join_agent = None
        self.query_agent = None
        self.validation_agent = SQLValidationAgent(connection, logger)
        self.evaluation_agent = QueryEvaluationAgent(connection, logger)
        self.supervisor = SupervisorAgent(logger)
        self.department = "Orchestrator"

    def process_user_query(self, user_query):
        try:
            self.schema = self.schema_agent.extract_schema()
            if not self.schema:
                raise ValueError("Schema extraction returned empty.")

            self.join_agent = JoinDeterminationAgent(self.schema, self.logger)
            join_info = self.join_agent.determine_joins(user_query)

            self.query_agent = QueryGenerationAgent(self.schema, self.logger)
            sql_query = self.query_agent.generate_query(user_query, join_info)

            valid, error = self.validation_agent.validate_query(sql_query)
            if not valid:
                raise ValueError(f"SQL validation failed: {error}")

            evaluation = self.evaluation_agent.evaluate_query(sql_query)
            supervisor_errors = self.supervisor.review_discussions()

            self.logger.log(self.department, "Query processing completed successfully.")
            return sql_query, evaluation, supervisor_errors
        except Exception as e:
            self.logger.log(self.department, f"Error in processing query: {e}", "ERROR")
            return None, None, [str(e)]

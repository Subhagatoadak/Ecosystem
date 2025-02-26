import os
import sqlite3
import logging
from Agents.SQL.SQLAgent import SQLQueryOrchestrator, DiscussionLoggerAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    # Connect to your SQLite database (path set via SQLITE_DB_PATH environment variable)
    db_path = os.getenv("SQLITE_DB_PATH", "example.db")
    try:
        connection = sqlite3.connect(db_path)
        logging.info(f"Connected to database: {db_path}")
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        return

    # Initialize the discussion logger and orchestrator.
    logger = DiscussionLoggerAgent()
    orchestrator = SQLQueryOrchestrator(connection, logger)

    # Get the natural language query from the user.
    user_query = input("Enter your natural language query: ")

    # Process the query through all agents.
    sql_query, evaluation, supervisor_errors = orchestrator.process_user_query(user_query)
    if sql_query:
        print("\nFinal SQL Query:")
        print(sql_query)
        print("\nQuery Evaluation/Optimization Plan:")
        print(evaluation)
        print("\nDiscussion Log:")
        for log in logger.get_logs():
            print(log)
    else:
        print("Query processing failed with errors:")
        for error in supervisor_errors:
            print(error)

if __name__ == "__main__":
    main()

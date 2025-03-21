import duckdb
import pandas as pd
import google.generativeai as genai
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import atexit
from dotenv import load_dotenv
import re
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt, matplotlib
import io
import base64
import random
import pdfplumber
from openai import OpenAI

# Initialize Flask App
app = Flask(__name__)


# Enable CORS
CORS(app)

load_dotenv()

# Set API Keys
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")  # Replace with your MotherDuck token
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")      # Replace with your Google Gemini API Key
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")       # Replace with your OpenAI API Key

# Configure Google Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Set the backend for Matplotlib to Agg
matplotlib.use("Agg")

# Connect to MotherDuck
conn = duckdb.connect(database=f"md:my_db?motherduck_token={MOTHERDUCK_TOKEN}")
conn.execute(f"INSTALL motherduck; LOAD motherduck;")

# Store uploaded tables for deletion on shutdown
uploaded_tables=[]
metadata_store = {}

@app.route("/")
def home():
    return render_template("index.html")

# Upload a dataset to MotherDuck from a CSV file
@app.route("/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    files = request.files.getlist("file")
    
    
    if not files:
        return jsonify({"error": "No selected file"}), 400

    for file in files:
        try:
            table_name = request.form.get("table_name", file.filename.split('.')[0])  # Use filename without extension
            table_name = re.sub(r'\W+', '_', table_name)  # Replace special characters with underscores

            # Check file extension and load dataset accordingly
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')  # Handle potential encoding issues
            elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                df = pd.read_excel(file)
            elif file.filename.endswith('.json'):
                df = pd.read_json(file)
            elif file.filename.endswith(".txt"):
                df = pd.DataFrame({"text": file.read().decode("utf-8").splitlines()})
            elif file.filename.endswith(".pdf"):
                with pdfplumber.open(file) as pdf:
                    text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                    df = pd.DataFrame({"text": [text]})
            elif file.filename.endswith(".png") or file.filename.endswith(".jpg"):
                image = Image.open(file)
                text = pytesseract.image_to_string(image)
                df = pd.DataFrame({"text": [text]})
            else:
                return jsonify({"error": "Unsupported file format. Supported formats: CSV, Excel, JSON, TXT, PDF, Images"}), 400
            print(df)
            # Register the table in MotherDuck
            conn.register("temp_table", df)
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM temp_table;")
            print("check")
            # Store metadata
            metadata = get_metadata(df)
            uploaded_tables.append(table_name)
            metadata_store[table_name] = metadata

            print(f"✅ Dataset '{table_name}' uploaded and stored in MotherDuck.")

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    print(f"📊 Uploaded tables: {uploaded_tables}")
    # Convert list to a comma-separated string
    uploaded_tables_str = ', '.join(uploaded_tables)

    # Return metadata and success message to frontend
    return jsonify({"success": True, "message": f"{uploaded_tables_str}"}), 200

# Convert a natural language query into SQL
def generate_sql_query(nl_query: str):
    
    prompt = f"""
    You are an expert SQL assistant working with DuckDB.
    Convert the following natural language query into a SQL query that works on the tables':
    table information: {metadata_store},
    Query: "{nl_query}"

    Guidelines:
    - If the query is related to the dataset, generate a valid SQL query that I can directly execute on dataset.
    - Use the correct column names and table names from the dataset provided.
    - If the query is not relevant to the dataset (e.g., personal questions, general knowledge, related to other dataset not mentioned here, non-related question or location-based queries), return the following message:
      ```
        None
      ```
    - Ensure the SQL syntax is valid for DuckDB.
    - 
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    # Clean the query by removing markdown syntax 
    return response.text.replace("```sql", "").replace("```", "").strip()

# Executes the SQL query on MotherDuck and returns the results.
def execute_sql_query(sql_query: str):
    try:
        result = conn.execute(sql_query).fetchdf()
        return result.to_html(classes="styled-table", escape=False)  # Convert to HTML table
    except Exception as e:
        return {"error": str(e)}, 500


# Extract metadata from a dataset
def get_metadata(data):
    return {
        "columns": data.columns.tolist(),
        "dtypes": data.dtypes.astype(str).to_dict(),
        "rows": len(data),
        "shape": data.shape,
        "summary": data.describe(include="all").to_dict()
    }

# Funny fallback function for unrelated queries
def funny_fallback():
    funny_sql_queries = [
        "SELECT 'I am not a philosopher, but I can query a database!' AS response;",
        "SELECT 'Error 404: Your question is too deep for me!' AS wisdom;",
        "SELECT 'Sorry, I only speak SQL, not riddles!' AS chatbot_confusion;",
        f"SELECT 'I have no idea where you are, but you are definitely online!' AS location_info;"
    ]

    funny_results = [
        "I'm an AI, not a detective! 🕵️‍♂️",
        "I tried asking the database, but it just shrugged. 🤷",
        "SQL only understands numbers and text, not existential crises! 😆",
        "If I had emotions, I'd be confused too. 😂",
        "Your question just crashed my humor module. Please reboot. 🔄"
    ]

    return {
        "sql_query": random.choice(funny_sql_queries),
        "results":  {"error": random.choice(funny_results)}
    }

# Detect intent in a user query using Gemini
def detect_intent_gemini(user_query):
    prompt = f"""
    You are an AI that detects intent in user queries. Categorize the given query into one of the following:
    - "sql_query" if the user wants a SQL result.
    - "visualization" if the user wants a graph, chart, or visualization.
    - "both" if the user wants both SQL and visualization.
    - "None" if the query is not relevant to the dataset or location-based queries or personal question.

    Query: "{user_query}"
    Intent:
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    intent = response.text.strip()
    return intent

# Generate Python code for a given natural language query
def generate_python_code(nl_query: str):
    prompt = f"""
    You are a Python developer working on a data visualization project.
    Convert the following natural language query into a Python code snippet that generates a visualization using Matplotlib:
    Query: "{nl_query}"
    Dataset Information: "{metadata_store}"
    Guidelines:
    - Use the correct column names and table names from the dataset provided.
    - Ensure the Python code is valid and can generate a visualization using Matplotlib.
    - If the query is not relevant to the dataset (e.g., personal questions, general knowledge, related to other dataset not mentioned here, non-related question or location-based queries), return the following message:
      ```
        None
      ```
    - You can generate a bar chart, line chart, pie chart, scatter plot, or any other relevant visualization.
    - 
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    # Clean the query by removing markdown syntax 
    return response.text.replace("```python", "").replace("```", "").strip()

def execute_generated_code(python_code):
    try:
        # Create an in-memory bytes buffer
        buffer = io.BytesIO()

        # Execute the generated code
        exec(python_code, globals())

        # Save the figure to the buffer instead of a file
        plt.savefig(buffer, format="png", bbox_inches="tight")
        # Encode the image to base64
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()  # Close the figure to free memory

        return img_base64
    except Exception as e:
        return str(e)

# API endpoint to convert a natural language query into SQL and execute it
@app.route("/query", methods=["POST"])
def query():
    data = request.json # Get the JSON data from the request
    nl_query = data["query"]  # Extract the natural language query 
    # Detect the intent of the user query
    intent = detect_intent_gemini(nl_query)
    print(intent)
    try:
        if intent == "visualization":
            python_code = generate_python_code(nl_query)
            if 'None' in python_code:
                return jsonify(funny_fallback())
            image_base64= execute_generated_code(python_code)
            return jsonify({ "image_base64": image_base64})
        
        elif intent == "both":
            sql_query = generate_sql_query(nl_query)
            if 'None' in sql_query:
                return jsonify(funny_fallback())
            results = execute_sql_query(sql_query)
            python_code = generate_python_code(nl_query)
            image_base64= execute_generated_code(python_code)
            return jsonify({"sql_query": sql_query, "results": results,  "image_base64":image_base64})
        
        elif intent == "sql_query":
            sql_query = generate_sql_query(nl_query)
            if 'None' in sql_query:
                return jsonify(funny_fallback())
            results = execute_sql_query(sql_query)
            return jsonify({"sql_query": sql_query, "results": results})
        
        else:
            print("dsa")
            return jsonify(funny_fallback())
        
    except Exception as e:
        return jsonify({"error": str(e)})

# Delete all uploaded tables when the server shuts down
def delete_uploaded_tables():
    if conn:
        try:
            for table in uploaded_tables:
                conn.execute(f"DROP TABLE IF EXISTS {table};")
                print(f"🗑️ Table '{table}' deleted from MotherDuck.")
        except Exception as e:
            print(f"⚠️ Error deleting tables: {e}")
        finally:
            conn.close()  # Close the connection properly
            print("🔌 Database connection closed.")

# Register cleanup function on server exit
atexit.register(delete_uploaded_tables)

# Add new endpoint to get table list
@app.route("/get_tables", methods=["GET"])
def get_tables():
    try:
        # Query DuckDB for all tables in the main schema
        result = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main';").fetchdf()
        tables = result['table_name'].tolist()
        return jsonify({"tables": tables})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
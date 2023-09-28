from flask import Flask, render_template, request, redirect, session, jsonify
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)

# from llama_cpp import Llama
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  
from transformers import GPT2LMHeadModel, GPT2Tokenizer  
import torch
import psycopg2
import os
import subprocess
import dbChat
from datetime import datetime, timezone
import secrets
import shutil
import re
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from constants import PERSIST_DIRECTORY, DEVICE_TYPE, SHOW_SOURCES





app = Flask(__name__)
app.secret_key = "your_secret_key"

app.databse_info=""

# List to store queries and responses
queries = {}
queries[""] = []

user_data = {}


def save_text_content(url, save_path):
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--incognito')
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(options=chrome_options)
    width = 1080
    height = 1920
    driver.set_window_size(height, width)
    driver.get(url)
    
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace(".", "_")
    path = parsed_url.path.strip("/").replace("/", "_")
    file_name = domain + "_" + path + ".html"
    
    title = driver.title
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    header = soup.find('header')
    if header:
        header.extract()
    nav = soup.find('nav')
    if nav:
        nav.extract()
    footer = soup.find('footer')
    if footer:
        footer.extract()
    for script in soup(["script", "style"]):
        script.extract()
        
    text_content = soup.get_text().strip()
    text_content = ' '.join(text_content.split())
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <title>{title}</title>
    </head>
    <body>
    <p>{text_content}</p>
    </body>
    </html>
    """
    html_save_path = save_path + "/" + file_name
    with open(html_save_path, "w", encoding='utf-8') as f:
        f.write(html_content)
    driver.close()
    return file_name

def load_user_data(username):
    data_path = f"queries/{username}/queries.json"
    if os.path.isfile(data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
            user_data.update({username: data})
    else:
        data = {}
        os.makedirs(os.path.dirname(data_path), exist_ok=True)  # Create directories if they don't exist
        with open(data_path, "w") as f:
            json.dump(data, f)
            
def save_user_data(username):
    data_path = f"queries/{username}/queries.json"
    data = user_data[username]
    if os.path.isfile(data_path):
        with open(data_path, "w") as f:
            json.dump(data, f)

users = os.listdir("prefix_SOURCE_DOCUMENTS/")
for user in users:
    load_user_data(user)

            
def process_response(response):
    expression = "> source_document"
    substrings = re.split(expression, response)
    
    # Add prefix to each element
    for i in range(len(substrings)):
        if i == 0:
            substrings[i] = "Starting: " + substrings[i]
        else:
            substrings[i] = expression + substrings[i]
    
    return substrings

def get_db_info():
    db_info=''
    conn = get_db_connection()
    
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'")

    # Fetch all results as a list of tuples
    table_names = cursor.fetchall()
    for table in table_names:
        cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = '"+table[0]+"'")
        column_names = [result[0] for result in cursor.fetchall()]
        my_string = ', '.join(column_names)
        db_info+= "\n table name:"+table[0]+", columes:"+my_string
        
    print(db_info)  
    return db_info              
def get_db_connection():
    
    connection = psycopg2.connect(
        host="localhost", port="5432", dbname="knowledge_maker", user="postgres", password="allow me"
    )
    # print(connection)
    return connection


@app.before_request
def before_request():
    if request.endpoint == "login" and "user_id" in session:
        return redirect("/home")


@app.route("/delete_project", methods=["POST"])
def delete_project():
    if "user_id" in session:
        # Get the project_name from the request
        project_name = request.form["project_name"]
        if project_name == session["project_name"]:
            session["project_name"] = ""

        # Retrieve the username from the session
        username = session.get("username")

        user_data[username].pop(project_name)
        save_user_data(username)

        # Specify the folder path where the projects are saved
        folder_path = "prefix_SOURCE_DOCUMENTS/" + username

        # Build the full path of the directory to delete
        project_path = os.path.join(folder_path, project_name)

        try:
            # Check if the directory exists
            if os.path.isdir(project_path):
                # Delete the directory and its contents
                shutil.rmtree(project_path)
        except Exception as e:
            # Handle any errors that occur during directory deletion
            pass

    return redirect("/home")  # Redirect to the home page


@app.route("/delete", methods=["POST"])
def delete_file():
    if "user_id" in session:
        # Get the filename from the request
        filename = request.form["filename"]

        # Retrieve the username from the session
        username = session.get("username")
        project_name = session.get("project_name")

        # Specify the folder path where the uploaded files are saved
        folder_path = "prefix_SOURCE_DOCUMENTS/" + username + "/" + project_name

        # Build the full path of the file to delete
        file_path = os.path.join(folder_path, filename)

        try:
            # Check if the file exists
            if os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)
        except Exception as e:
            # Handle any errors that occur during file deletion
            pass

    return redirect("/home")  # Redirect to the home page


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        print(username+": "+ password)
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            "SELECT id, password FROM tb_users WHERE username = %s", (username,)
        )
        user = cursor.fetchone()
        if user and user[1] == password:
            session["user_id"] = user[0]
            session["username"] = username
            session["start_time"] = datetime.now(timezone.utc)
            session["session_id"] = secrets.token_hex(16)  # Generate a session ID
            session["project_name"] = ""
            print(session)
            # Create the "templates/mitta" folder if it doesn't exist
            folder_path = os.path.join("prefix_SOURCE_DOCUMENTS", username)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            folder_path = os.path.join("queries", username)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Insert the session start time and username into the sessions table
            time_start = session["start_time"]
            time_start_str = time_start.strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT INTO sessions (session_id, username, time_start) VALUES (%s, %s, %s)",
                (session["session_id"], username, time_start_str),
            )
            connection.commit()
            cursor.close()
            connection.close()
            return redirect("/home")
        else:
            error = "Invalid username or password"
            return render_template("login.html", error=error)

        

    return render_template("login.html")


@app.route("/home")
def home():
    if "user_id" in session:
        # app.database_info = "This is the Postgres database information" + get_db_info()
        # print(app.database_info)
        # Retrieve the username from the session
        username = session.get("username")
        load_user_data(username)
        project_name = session.get("project_name")
        # Specify the folder path where the uploaded files will be saved
        folder_path = "prefix_SOURCE_DOCUMENTS/" + username

        # Get the list of projects
        projects = user_data[username].keys()
        queries = []
        if project_name:
            queries = user_data[username][project_name]["queries"]
            
        folder_path += "/" + project_name
        files = os.listdir(folder_path)
        file_names = []
        for file in files:
            if os.path.isfile(os.path.join(folder_path, file)):
                file_names.append(file)

        return render_template(
            "index.html",
            projects=projects,
            files=file_names,
            queries=queries,
            username=username,
            project_name=project_name,
        )
    else:
        return redirect("/login")


@app.route("/")
def index():
    if "user_id" in session:
        return redirect("/home")
    else:
        return redirect("/login")

@app.route("/logout")
def logout():
    if "user_id" in session:
        connection = get_db_connection()
        cursor = connection.cursor()

        session_id = session.get("session_id")
        time_start = session.get("start_time")
        time_end = datetime.now(timezone.utc)

        time_end_str = time_end.strftime("%Y-%m-%d %H:%M:%S")
        duration = time_end - time_start

        # Update the session end time and duration in the sessions table
        cursor.execute(
            "UPDATE sessions SET time_end = %s, duration = %s WHERE session_id = %s",
            (time_end_str, duration, session_id),
        )
        connection.commit()

        cursor.close()
        connection.close()

    session.clear()
    return redirect("/login")

@app.route("/webupload", methods=["POST"])
def web_upload():
    url = request.form.get('text')
    username = session.get("username")
    project_name = session.get("project_name")
    # Specify the folder path where the uploaded files will be saved
    folder_path = "prefix_SOURCE_DOCUMENTS/" + username

    # Check if the project folder exists, create it if it doesn't
    project_folder_path = folder_path + "/" + project_name
    if not os.path.exists(project_folder_path):
         os.makedirs(project_folder_path)
    filename = save_text_content(url, project_folder_path)
    
    timestamp = datetime.now(timezone.utc)
    # Insert the document name, username, project name, and timestamp into the uploaded_documents table
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO uploaded_documents (document_name, username, project_name, timestamp) VALUES (%s, %s, %s, %s)",
        (filename, username, project_name, timestamp),
    )
    connection.commit()
    cursor.close()
    connection.close()
    
     # Run the first Python program on each uploaded file
    python_file_path = "ingest.py"
    process = subprocess.Popen(
        ["python", python_file_path, username, project_name]
    )
    process.wait()
    return "done", 200

@app.route("/upload", methods=["POST"])
def upload():
    if "user_id" in session:
        username = session.get("username")
        project_name = session.get("project_name")
        # Specify the folder path where the uploaded files will be saved
        folder_path = "prefix_SOURCE_DOCUMENTS/" + username

        # Check if the project folder exists, create it if it doesn't
        project_folder_path = os.path.join(folder_path, project_name)
        if not os.path.exists(project_folder_path):
            os.makedirs(project_folder_path)

        # Get the uploaded file(s)
        uploaded_files = request.files.getlist("file")
        timestamp = datetime.now(timezone.utc)

        # Save the uploaded file(s) to the project folder
        for file in uploaded_files:
            file.save(os.path.join(project_folder_path, file.filename))

            # Insert the document name, username, project name, and timestamp into the uploaded_documents table
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO uploaded_documents (document_name, username, project_name, timestamp) VALUES (%s, %s, %s, %s)",
                (file.filename, username, project_name, timestamp),
            )
            connection.commit()
            cursor.close()
            connection.close()

        # Run the first Python program on each uploaded file
        python_file_path = "ingest.py"
        process = subprocess.Popen(
            ["python", python_file_path, username, project_name]
        )
        process.wait()

        return redirect("/home")
    else:
        return redirect("/login")


@app.route("/summary", methods=["POST"])
def summary():
    if "user_id" in session:
        username = session.get("username")
        project_name = session.get("project_name")
        file_path = request.form["filename"]
        print("Summarizing file: ", file_path)

        # Run the second Python program and capture its output
        python_file_path = "summarizer.py"
        process = subprocess.Popen(
            [
                "python",
                python_file_path,
                username,
                project_name,
                file_path,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        output, _ = process.communicate()

        process.wait()

        trimmed_output = output.split(">>>summary>>>")[1]
        out_response = {"type": "Summary for: ", "query": file_path, "response": trimmed_output, "sources": []}

        print(out_response)

        # Add summary
        user_data[username][project_name]["queries"].append(out_response)
        save_user_data(username)

        return jsonify(out_response)
    else:
        return redirect("/login")



@app.route("/query", methods=["POST"])
def query():
    if "user_id" in session:
        # Get the user query from the AJAX request
        user_query = request.form["query"]
        username = session.get("username")
        project_name = session.get("project_name")
        
        # # Run the second Python program and capture its output
        python_file_path = "privateGPT.py"
        process = subprocess.Popen(
            [
                "python",
                python_file_path,
                "--user",
                username,
                "--project",
                project_name,
                "--query",
                user_query,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        output, _ = process.communicate()

        process.wait()
        trimmed_output = output.split(">>>response>>>")[1].strip()
        print(trimmed_output)
        out_response = json.loads(trimmed_output)
        if not "error" in out_response:       
            user_data[username][project_name]["queries"].append({"type": "Query", "query": user_query, "response": out_response["response"], "sources": out_response["sources"]})
            save_user_data(username)
        else:
            print(out_response["error"])
            out_response = {"response": out_response["error"], "sources": []}
    
        return jsonify(out_response)
    else:
        return redirect("/login")


@app.route("/create_project", methods=["POST"])
def create_project():
    if "user_id" in session:
        username = session.get("username")
        project_name = request.form["project_name"]
        session["project_name"] = project_name
        user_data[username].update({project_name: {"queries": [], "summaries": {}}})
        save_user_data(username)

        # Specify the folder path where the uploaded files will be saved
        folder_path = "prefix_SOURCE_DOCUMENTS/" + username

        # Check if the project folder exists, create it if it doesn't
        project_folder_path = os.path.join(folder_path, project_name)
        if not os.path.exists(project_folder_path):
            os.makedirs(project_folder_path)

        return redirect("/home")
    else:
        return redirect("/login")


@app.route("/select_project", methods=["POST"])
def select_project():
    if "user_id" in session:
        project_name = request.form["project_name"]
        session["project_name"] = project_name

        return redirect("/home")
    else:
        return redirect("/login")


@app.route("/reset", methods=["POST"])
def reset_queries():
    if "user_id" in session:
        username = session["username"]
        project_name = session["project_name"]
        user_data[username][project_name] = {"queries": []}
        save_user_data(username)
        return redirect("/home")
    else:
        return redirect("/login")


@app.route("/chat_data")
def chat_data():
    if "user_id" in session:
        # Retrieve the username from the session
        username = session.get("username")
        return render_template("chat_data.html")
    else:
        return redirect("/login")

@app.route("/upload_db", methods=["POST"])
def upload_db():
    if "user_id" in session:
        # Retrieve the username from the session
        username = session.get("username")
        if request.method == 'POST':  
            f = request.files['file']
            data = f.read()
            f.save("db/"+"db.txt")  
            # python_file_path = "dbChat.py"
            # process = subprocess.Popen(
            #     [
            #         "python",
            #         python_file_path,
            #         "--user",
            #         username,
            #         "--db_data",
            #         data
            #     ],
            #     stdout=subprocess.PIPE,
            #     text=True,
            # )
            # output, _ = process.communicate()
            
            print(data)
        data = {'message':'Hello'}
        return jsonify(data)
    else:
        return redirect("/login")

@app.route("/textTosql", methods=["POST"])
def textTosql():
    if "user_id" in session:
        # Retrieve the username from the session
        username = session.get("username")
        model = T5ForConditionalGeneration.from_pretrained('dsivakumar/text2sql')
        tokenizer = T5Tokenizer.from_pretrained('dsivakumar/text2sql')
        def get_sql(query,tokenizer,model):
            source_text= "English to SQL: "+query
            source_text = ' '.join(source_text.split())
            source = tokenizer.batch_encode_plus([source_text],max_new_tokens= 128, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
            source_ids = source['input_ids'] #.squeeze()
            source_mask = source['attention_mask']#.squeeze()
            generated_ids = model.generate(
                input_ids = source_ids.to(dtype=torch.long),
                attention_mask = source_mask.to(dtype=torch.long), 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            return preds
        print(get_sql("Show me all users that name is areeb", tokenizer=tokenizer,model=model));
    else:
        return redirect("/login")

@app.route("/textTosql1", methods=["POST"])
def textTosql1():
    if "user_id" in session:
        get_db_info()
        
        return "OK";
    else:
        return redirect("/login")

@app.route("/generate", methods=["POST"])
def generate():
    if "user_id" in session:
        prompt = request.form["prompt"]
        username = session.get("username")
        project_name = session.get("project_name")
        db_info=get_db_info()
        #Interact with model
        res=dbChat.generate(db_info,prompt)
        print(res)
        words = res.split()
        # find the index of the word "FROM"
        from_index = words.index("FROM")
        # extract the table name, which should be the word after "FROM"
        table_name = words[from_index + 1]
        
        print("Got the result____")
        # print(res)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = '"+table_name+"'")
        column_names = [result[0] for result in cursor.fetchall()]
        cursor.close()
        cursor=conn.cursor()
        cursor.execute(res)
        results = cursor.fetchall()
        print(column_names)
        data = {'column_names': column_names, 'results': results}
        print(data)
        # print(results)
        # print("Start chatting..."+output)
        # res =await dbChat.chat(prompt)
        print("End...")
        return data
    else:
        return redirect("/login")
if __name__ == "__main__":
    app.run(debug=True)

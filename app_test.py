from flask import Flask, render_template, request, redirect, session
import psycopg2
import os
import subprocess
from datetime import datetime, timezone
import secrets
import shutil

app = Flask(__name__)
app.secret_key = "your_secret_key"

# List to store queries and responses
queries = {}
queries[""] = []


def save_queries_to_file(username):
    with open(f"queries/{username}/queries.txt", "w") as file:
        for project_name, project_queries in queries.items():
            file.write(f"Project: {project_name}\n")
            for query_response in project_queries:
                query = query_response["query"]
                response = query_response["response"]
                file.write(f"Query: {query}\n")
                file.write(f"Response: {response}\n")
            file.write("\n")


def load_queries_from_file(username):
    project_name = ""
    query = ""
    response = ""
    if os.path.isfile(f"queries/{username}/queries.txt"):
        queries.clear()
        queries[""] = []
        with open(f"queries/{username}/queries.txt", "r") as file:
            is_query = False
            is_response = False
            for line in file:
                line = line.strip()
                if line.startswith("Project:"):
                    if response.startswith("("):
                        queries[project_name].append(
                            {"query": query, "response": response}
                        )
                    query = ""
                    response = ""
                    project_name = line[9:]
                    queries[project_name] = []
                    is_query = False
                    is_response = False
                elif (
                    line.startswith("Response:") or is_response
                ) and not line.startswith("Query:"):
                    is_response = True
                    if line.startswith("Response:"):
                        response += line[10:]
                    else:
                        response += line
                elif line.startswith("Query:") or is_query:
                    if response.startswith("("):
                        queries[project_name].append(
                            {"query": query, "response": response}
                        )
                    response = ""
                    query = ""
                    is_query = True
                    if line.startswith("Query:"):
                        query += line[7:]
                    else:
                        query += line
            if response.startswith("("):
                queries[project_name].append({"query": query, "response": response})
            print(queries)


def get_db_connection():
    connection = psycopg2.connect(
        host="localhost", port="5432", dbname="knowledge_maker", user="postgres", password="hitr"
    )
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

        queries.pop(project_name)
        save_queries_to_file(username)

        # Specify the folder path where the projects are saved
        folder_path = "source_documents/" + username

        # Build the full path of the directory to delete
        project_path = os.path.join(folder_path, project_name)

        try:
            # Check if the directory exists
            if os.path.isdir(project_path):
                # Delete the directory and its contents
                shutil.rmtree(project_path)
        except Exception as e:
            # Handle any errors that occur during directory deletion
            print(f"Failed to delete project directory: {e}")

        return redirect("/home")


@app.route("/upload", methods=["POST"])
def upload():
    if "user_id" in session:
        # Retrieve the username from the session
        username = session.get("username")
        # Retrieve the project name from the session
        project_name = session.get("project_name")

        # Check if a file was uploaded
        if "file" not in request.files:
            return redirect("/home")

        file = request.files["file"]

        # Check if the file is empty
        if file.filename == "":
            return redirect("/home")

        # Specify the folder path where the projects are saved
        folder_path = "source_documents/" + username

        # Create the folder if it doesn't exist
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        # Build the full path of the destination file
        file_path = os.path.join(folder_path, project_name, file.filename)

        # Save the file to the destination path
        file.save(file_path)

        return redirect("/home")


@app.route("/home")
def home():
    if "user_id" in session:
        # Retrieve the username from the session
        username = session.get("username")
        # Retrieve the project name from the session
        project_name = session.get("project_name")

        # Specify the folder path where the projects are saved
        folder_path = "source_documents/" + username

        # Get the list of projects for the current user
        projects = os.listdir(folder_path) if os.path.isdir(folder_path) else []

        return render_template(
            "home.html",
            username=username,
            project_name=project_name,
            projects=projects,
        )
    return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Perform authentication
        # ...

        # Store the username in the session
        session["username"] = username
        session["project_name"] = ""

        # Create the folder for the user's queries if it doesn't exist
        queries_folder = f"queries/{username}"
        if not os.path.isdir(queries_folder):
            os.makedirs(queries_folder)

        # Load the user's queries from the file
        load_queries_from_file(username)

        return redirect("/home")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("project_name", None)
    return redirect("/login")


@app.route("/projects", methods=["POST"])
def select_project():
    if "user_id" in session:
        # Retrieve the selected project name from the request
        project_name = request.form["project_name"]

        # Store the project name in the session
        session["project_name"] = project_name

        return redirect("/home")

    return redirect("/login")


@app.route("/")
def index():
    return redirect("/login")


if __name__ == "__main__":
    app.run(debug=True)

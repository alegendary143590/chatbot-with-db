pip install -r requirements.txt

download the LLM model and place it in a models folder:
- LLM: default to [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin) or huggingface.

Folder name "LaMini-Flan-T5-248M" and download each files from huggingface of this model and save in this folder.(https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M/tree/main)

The supported extensions are:

   - `.csv`: CSV,
   - `.docx`: Word Document,
   - `.doc`: Word Document,
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.html`: HTML File,
   - `.md`: Markdown,
   - `.msg`: Outlook Message,
   - `.odt`: Open Document Text,
   - `.pdf`: Portable Document Format (PDF),
   - `.pptx` : PowerPoint Document,
   - `.ppt` : PowerPoint Document,
   - `.txt`: Text file (UTF-8),
   - `.xlsx`: excel file (xlsx),

use post greSQL with this credentials connection = psycopg2.connect
        host="localhost", port="5432", dbname="knowledge_maker", user="postgres", password="hitr"

then copy paste below details in query tool and run

-- DROP SCHEMA PUBLIC CASCADE;
-- CREATE SCHEMA PUBLIC;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL
);

CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    username VARCHAR(255) NOT NULL,
    time_start TIMESTAMP NOT NULL,
    time_end TIMESTAMP,
    duration INTERVAL
);

CREATE TABLE uploaded_documents (
    id SERIAL PRIMARY KEY,
    document_name VARCHAR(255) NOT NULL,
    username VARCHAR(255) NOT NULL,
    project_name VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL
);

INSERT INTO USERS(username, password) VALUES('hi','hi');

If you find difficulty in uploading MS word documents then please convert to pdf files then again upload documents.
    
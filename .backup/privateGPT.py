#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time
import json

from constants import CHROMA_SETTINGS

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
# persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))


def main():
    # Parse the command line arguments
    args = parse_arguments()
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    persist_directory = os.path.join(os.path.join(persist_directory, args.user),args.project)
    CHROMA_SETTINGS.persist_directory = os.path.join(os.path.join(
        CHROMA_SETTINGS.persist_directory, args.user),args.project)
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks,
                           verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch,
                          callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=not args.hide_source)

    # Check if query is provided as a command-line argument
    if args.query:
        query = args.query
        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [
        ] if args.hide_source else res['source_documents']
        end = time.time()
        
        out_response = {"query": query, "response": answer, "sources": []}
        
        # Print the relevant sources used for the answer
        for document in docs:
            out_response["sources"].append(document.metadata["source"] + ": " + document.page_content)
        
        # test format
        
        print(">>>response>>>" + json.dumps(out_response))


def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    parser.add_argument("--query", type=str,
                        help='The query to ask the model.')

    parser.add_argument("--user", type=str,
                        help='Username trying to accss.')
    
    parser.add_argument("--project", type=str,
                        help='Project user is trying to accss.')

    return parser.parse_args()


if __name__ == "__main__":
    main()

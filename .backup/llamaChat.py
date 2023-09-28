from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

from constants import CHROMA_SETTINGS

DB_FAISS_PATH = 'vectorstore/db_faiss'
persist_directory = "db/hi/test"

custom_prompt_template = """SYSTEM: You are a helpful, respectful and honest assistant. 
Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful,
unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are 
socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, 
say you don't know the answer and explain why instead of answering something not correct. If you don't know the answer to a question, 
please don't share false information.
- CONTEXT: {context}
- USER: {question}
- ASSISTANT:"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        temperature = 0.7,
        top_k = 50,
        top_p = 0.9,
        repetition_penalty = 1.0,
        max_new_tokens = 256,
        seed = 42
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    #db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    CHROMA_SETTINGS.persist_directory = persist_directory
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

print(final_result("What is the relation of India and berlin ?"))

# load the large language model file

from llama_cpp import Llama
LLM = Llama(model_path="./models/llama-2-7b-chat.ggmlv3.q8_0.bin")

def generate(db_info,query):
    prompt = "Q:This is Postgres database information: {table name: Users, columes:id, name, password, address, age}. For this database, English to SQL: "
    suffix =".Only output the sql query without any explanation.For instance, if the input is find all users logged in this week, the output should be SELECT * FROM sessions WHERE time_start::timestamp >= date_trunc('week', CURRENT_TIMESTAMP);. A:"
    preffix = "Q:This is Postgres database information: {"
    middle = "}. For this database, English to SQL: "
    prompt=preffix+db_info+middle+query+suffix
    print("Generating..."+query)
    output = LLM(prompt)
    # display the response
    return output["choices"][0]["text"]
# def main(query):

#     print("ok")
# if __name__ == "__main__":
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
#     )
    
    
#     try:
#         start = time.time()
#         res = main(query)
#         end = time.time()
#         total_time = round(end - start, 2)
#         # final_res = {"response": f"Took({total_time})s:" + res["result"], "sources": []}

#     except Exception as e:
#             print(">>>response>>>" + json.dumps({"error": str(e)}))

 
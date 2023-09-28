from llama_cpp import Llama
llm = Llama(model_path="./llama-2-7b-chat.ggmlv3.q8_0.bin")
output = llm("Q: tesla Quaterly result summary for 2023? A: ", max_tokens=100, stop=["Q:", "\n"], echo=True)
print(output)
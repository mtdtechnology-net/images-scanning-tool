from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# Inițializăm modelul. Poți schimba 'temperature' (0 = strict, 1 = mai creativ)
llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)
chat_history = []

print("=== Qwen2.5:7b Chat ===")
print("Scrie 'exit' sau 'quit' pentru a ieși.\n")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("La revedere!")
            break
            
        if not user_input.strip():
            continue
            
        chat_history.append(HumanMessage(content=user_input))
        
        print("Qwen: ", end="", flush=True)
        full_response = ""
        
        # Folosim stream() ca să afișeze textul literă cu literă (ca ChatGPT), nu să aștepte tot răspunsul
        for chunk in llm.stream(chat_history):
            print(chunk.content, end="", flush=True)
            full_response += chunk.content
            
        print("\n" + "-" * 50)
        
        # Salvăm răspunsul în istoric pentru ca modelul să țină minte contextul conversației
        chat_history.append(AIMessage(content=full_response))
        
    except KeyboardInterrupt:
        # Prinde Ctrl+C fără să crape urât
        print("\nLa revedere!")
        break
    except Exception as e:
        print(f"\n[Eroare]: {e}")

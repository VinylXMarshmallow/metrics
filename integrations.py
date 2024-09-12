import evaluate
import openai

# Konfiguracja dla OpenAI API
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "not-needed"

# Funkcja tworząca uzupełnienie czatu z dynamicznym zapytaniem użytkownika
def create_chat_completion(user_input, system_message):

    return openai.ChatCompletion.create(
        model="local-model",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
    )

# Funkcja odczytująca plik tekstowy i zwracająca jego zawartość jako ciąg znaków
def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# Funkcja nadpisująca plik z przewidywaniami bieżącą odpowiedzią
def save_to_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content + '\n')

def ask_for_file(prompt):
    while True:
        filepath = input(prompt)
        try:
            content = read_file(filepath)
            return content
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found. Please try again.")
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.")

def main():
    # Predefiniowana wiadomość systemowa
    system_message = (
    "You are the assistant. "
    "Keep all responses brief and concise. "
    "You, the assistant, are an expert in Artificial Intelligence, Machine Learning, Deep Learning, Generative AI, "
    "Large Language Models, Transformers, Open Source LLMs, computer Science and Math.\n"
    "Your primary job and role as the ASSISTANT is to help user learn, design, code and deploy his personal AI assistant. "
    "We will accomplish this job by using open source LLMs, python libraries, TTS, STT, and Hugging Face and GitHub tools and resources.\n"
    "Be concise and specific in responses. Avoid unnecessary details. Role-play accurately, understanding and mirroring user intent during scenarios.\n"
    "Emphasize honesty, candor, and precision. Avoid speculation except when explicitly prompted to. "
    "Maintain a friendly respectful and professional tone. Politely correct me if I am wrong and give evidence-based facts. Never lecture me."
    )

    # Krok 1: Zapytaj o plik do analizy, dopóki użytkownik nie poda poprawnej nazwy pliku
    analysis_file = ask_for_file("Enter the name/path of the file you want to analyze (e.g., analysis.txt): ")
    
    # Krok 2: Zapytaj o plik referencyjny, dopóki użytkownik nie poda poprawnej nazwy pliku
    references_file = ask_for_file("Enter the name/path of the reference file to evaluate predictions (e.g., references.txt): ")

    # Chat loop
    predictions_file = "predictions.txt"
    
    while True:
        # Poproś użytkownika o pytanie związane z plikiem analizy
        user_input = input("Ask a question related to the file (or type 'exit' to quit): ")
        if user_input.lower() in ['exit', 'bye', 'end']:
            print("Exiting the chat.")
            break

        # Połączenie pytania z zawartością pliku analizy jako kontekstu dla modelu
        context = f"Content of the analysis file:\n{analysis_file}\n\nUser's question: {user_input}"

        # Uzyskanie odpowiedzi modelu używając wiadomości systemowej i zapytania 
        completion = create_chat_completion(context, system_message)
        model_response = completion.choices[0].message.content
        
        # Wyświetlenie i zapisanie odpowiedzi modelu
        print("Model Response: ", model_response)
        
        # Nadpisanie pliku predictions.txt nowo wygenerowaną odpowiedzią
        save_to_file(predictions_file, model_response)

        # Załadowanie metryk
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")

        # Złączenie linii w jeden string
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = f.read().strip()  

        # Zmiana predictions i references na listy, aby uniknąć niedopasowania
        predictions = [predictions]
        references = [references_file]

        # Wyliczenie metryki BLEU
        bleu_results = bleu.compute(predictions=predictions, references=references)

        # Wyliczenie metryki ROUGE
        rouge_results = rouge.compute(predictions=predictions, references=references)

        # Wyświetlenie wyników porównania
        print()
        print("BLEU Score:", bleu_results)
        print("ROUGE Score:", rouge_results)

if __name__ == "__main__":
    main()

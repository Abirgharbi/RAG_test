from src.answer_generator import ask

while True:
    query = input("\n Question: ")

    if query.lower() in ["exit", "quit"]:
        break

    response = ask(query)

    print("\n Réponse:\n")
    print(response)

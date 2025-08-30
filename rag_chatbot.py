import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import json

# --- 1. RAG System Setup ---
class RAGSystem:
    def __init__(self, data_file):
        """Initializes the RAG system by loading data and building a searchable index."""
        print("Bot: Loading knowledge base...")
        self.df = pd.read_csv(data_file)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = self._create_index()
        print("Bot: Knowledge base loaded. RAG system is ready.")

    def _create_index(self):
        """Creates numerical representations (embeddings) of the drug interactions and builds the FAISS index for fast searching."""
        self.df['description_for_search'] = self.df.apply(
            lambda row: f"The interaction between {row['drug_a']} and {row['drug_b']}", axis=1
        )
        embeddings = self.encoder.encode(self.df['description_for_search'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))
        return index

    def search(self, query, k=1):
        """Searches the index for the most relevant interaction based on the user's query."""
        query_embedding = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        best_match_data = self.df.iloc[indices[0][0]].to_dict()
        distance_score = distances[0][0]
        return best_match_data, distance_score

# --- 2. LM Studio Integration ---
def ask_local_llm(context, question):
    """Handles standard drug-drug interaction queries."""
    prompt = f"""
    You are a cautious and helpful pharmacy assistant. Your task is to analyze the user's [QUESTION] and the retrieved [CONTEXT] and provide a safe, relevant answer.

    [CONTEXT]
    - Drug A: "{context['drug_a']}"
    - Drug B: "{context['drug_b']}"
    - Severity: "{context['severity']}"
    - Interaction Details: "{context['interaction']}"

    [QUESTION]
    "{question}"

    [INSTRUCTIONS]
    1.  **Relevancy Check:** First, compare the drugs in the [QUESTION] with Drug A and Drug B from the [CONTEXT].
    2.  **If they do NOT match**, the context is irrelevant. Respond with ONLY this exact sentence: "I could not find a specific interaction for that combination in my knowledge base. For safety, please consult a healthcare professional."
    3.  **If they DO match**, answer the user's question based *only* on the context. Start with an emoji (✅, ⚠️, 🚨) based on the severity.
    """
    return _send_to_llm(prompt)

def ask_symptom_llm(contexts, substance, symptom):
    """Handles queries about a symptom when a substance (e.g., alcohol) is also mentioned."""
    context_string = ""
    for i, ctx in enumerate(contexts, 1):
        drug = ctx['drug_a'] if substance.lower() != ctx['drug_a'].lower() else ctx['drug_b']
        context_string += f"\n[CONTEXT {i} - Regarding {drug}]\n- Interaction with {substance}: {ctx['interaction']}\n- Severity: {ctx['severity']}"

    prompt = f"""
    You are a very cautious pharmacy assistant. A user is asking what medicine they can take for a '{symptom}' after consuming '{substance}'.
    Provide a safe, informational summary about common treatments based ONLY on the provided contexts. You MUST NOT recommend a medicine.

    {context_string}

    [INSTRUCTIONS]
    1.  Start with a clear disclaimer that you cannot provide medical advice and the user must consult a doctor.
    2.  For each context provided, present the information about the interaction with '{substance}'. For example, say "Regarding Paracetamol: ..." or "Regarding Ibuprofen: ...".
    3.  End with a strong, final recommendation to speak with a healthcare professional.
    """
    return _send_to_llm(prompt)

def ask_symptom_only_llm(symptom, treatments):
    """Handles queries about a symptom when no other substance is mentioned."""
    treatments_string = ", ".join(treatments)
    prompt = f"""
    You are a very cautious AI health assistant. A user is asking what medicine to take for a '{symptom}'.
    Provide a safe, general, informational response. You MUST NOT give medical advice or recommend a specific medicine.

    [BACKGROUND INFORMATION]
    Common over-the-counter treatments for '{symptom}' include: {treatments_string}.

    [INSTRUCTIONS]
    1.  Start with a very clear disclaimer that you are an AI and cannot provide medical advice.
    2.  State that for a symptom like '{symptom}', some common over-the-counter options are available, such as {treatments_string}.
    3.  Briefly and factually state what each is generally used for (e.g., "Paracetamol is a pain reliever and fever reducer.").
    4.  Do NOT compare them or suggest one is better than another.
    5.  End with a strong recommendation that the user MUST consult a doctor or pharmacist for a proper diagnosis and treatment plan.
    """
    return _send_to_llm(prompt)

def _send_to_llm(prompt):
    """A helper function to send any prompt to the LM Studio API."""
    api_url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = { "model": "local-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1 }
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error connecting to LM Studio. Please make sure the server is running. Details: {e}"

# --- 3. Query Handling Logic ---
def handle_symptom_query(rag_system, substance, symptom):
    """Finds interactions for common symptom treatments with a given substance."""
    symptom_map = {
        "fever": ["Paracetamol", "Ibuprofen"],
        "headache": ["Paracetamol", "Ibuprofen", "Aspirin"],
        "pain": ["Paracetamol", "Ibuprofen", "Aspirin"]
    }
    if symptom not in symptom_map:
        return "I can only provide information for symptoms like fever, headache, or pain."

    relevant_contexts = []
    for treatment in symptom_map[symptom]:
        context, distance = rag_system.search(f"{treatment} and {substance}")
        if distance < 1.5 and (substance.lower() in context['drug_a'].lower() or substance.lower() in context['drug_b'].lower()):
             relevant_contexts.append(context)

    if not relevant_contexts:
        return f"I do not have specific information about taking common medications for {symptom} with {substance}. Please consult a doctor."

    return ask_symptom_llm(relevant_contexts, substance, symptom)

def handle_symptom_only_query(symptom):
    """Handles queries about a symptom when no other drug or substance is mentioned."""
    symptom_map = { "fever": ["Paracetamol", "Ibuprofen"], "headache": ["Paracetamol", "Ibuprofen", "Aspirin"], "pain": ["Paracetamol", "Ibuprofen", "Aspirin"] }
    if symptom not in symptom_map:
        return "I can only provide general information for symptoms like fever, headache, or pain. A doctor's consultation is necessary for other issues."
    return ask_symptom_only_llm(symptom, symptom_map[symptom])

# --- 4. Chatbot Main Loop ---
def run_rag_chatbot():
    """Main function to run the chatbot's interactive loop."""
    rag_system = RAGSystem('interactions.csv')
    print("\n--- 🤖 Offline AI Drug Interaction Chatbot ---")
    print("Ask about a drug combination (e.g., 'Warfarin and Aspirin')")
    print("Or a symptom (e.g., 'what can I take for a headache?')")

    while True:
        user_query = input("\nUser: ")
        if user_query.lower() == 'exit':
            print("Bot: Goodbye!")
            break

        # --- UPGRADED: 3-Branch Intent Detection Logic ---
        query_lower = user_query.lower()
        substance = 'Alcohol' if 'alcohol' in query_lower else None
        symptom = next((s for s in ['fever', 'headache', 'pain'] if s in query_lower), None)

        if substance and symptom:
            # Case 1: Symptom + Substance (e.g., "fever and alcohol")
            response = handle_symptom_query(rag_system, substance, symptom)
        elif symptom:
            # Case 2: Symptom ONLY (e.g., "I have a fever")
            response = handle_symptom_only_query(symptom)
        else:
            # Case 3: Default to drug-drug interaction
            context, _ = rag_system.search(user_query)
            response = ask_local_llm(context, user_query)
        
        print(f"Bot: {response}")

if __name__ == "__main__":
    run_rag_chatbot()


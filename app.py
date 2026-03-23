import os
from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq

app = Flask(__name__)

VECTORSTORE_PATH = "faiss_index_movies"


embeddings = CohereEmbeddings(
    model="embed-english-v3.0"
)


llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0
)

def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(
            VECTORSTORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    return None

vectorstore = load_vectorstore()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if not vectorstore:
        return jsonify({"error": "Database not found."}), 500
    
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        results = vectorstore.similarity_search(user_query, k=5)
        context = "\n\n".join(doc.page_content for doc in results)

        prompt = f"""
        You are a movie expert.
        QUESTION: Find the movie titled "{user_query}" and provide a short summary.
        If not found, say "I couldn't find that movie."
        CONTEXT: {context}
        """
        response = llm.invoke(prompt)
        return jsonify({"response": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
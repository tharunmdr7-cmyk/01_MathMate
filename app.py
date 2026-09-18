import os
from flask import Flask, jsonify, request, send_from_directory
from google import genai
from chatbot_config import SYSTEM_PROMPT

app = Flask(__name__, template_folder=".")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.1-flash-lite"

@app.get("/")
def home():
    return send_from_directory(".", "index.html")

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Please enter a question."}), 400
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config={"system_instruction": SYSTEM_PROMPT, "temperature": 0.3},
        )
        return jsonify({"reply": response.text or "I couldn't generate a response."})
    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({"error": "The AI service could not process your request. Please try again."}), 500

if __name__ == "__main__":
    app.run()

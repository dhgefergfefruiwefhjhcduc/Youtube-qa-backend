from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import RAGPipeline

app = Flask(__name__)
CORS(app)

pipelines = {}

@app.route("/")
def health():
    return "Server is running", 200

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    video_id = data.get("video_id")
    question = data.get("question")

    if not video_id or not question:
        return jsonify({"error": "Missing video_id or question"}), 400

    print(f"Received question for video ID: {video_id}")

    if video_id not in pipelines:
        print("Creating new pipeline...")
        pipelines[video_id] = RAGPipeline(video_id=video_id)

    bot = pipelines[video_id]

    answer = bot.run(question)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run()

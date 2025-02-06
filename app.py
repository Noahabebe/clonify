import os
import wave
import numpy as np
import ffmpeg
from flask import Flask, request, jsonify, send_from_directory
from gradio_client import Client, file
from tts.generate_tts import generate_tts_audio
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
from groq import Groq  # Groq integration

app = Flask(__name__)

# MongoDB Setup
client = MongoClient("mongodb://root:OT9Xh66yfE3wkLuiTv59zpt1dI96zEgXTk2VQb8EHM1yPOUKuhu5tZq7PKbHy2hV@wc4cw8ck4ocskgk0oww08w0c:27017/?directConnection=true")  
db = client["video_storage"]
fs = gridfs.GridFS(db)

# Groq Setup
groq_client = Groq()

# Local Directories for Processed Files
PROCESSED_DIR = 'static/processed_videos'
os.makedirs(PROCESSED_DIR, exist_ok=True)

client_gradio = Client("https://anhhayghen-musetalkv.hf.space/")

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    """Analyzes audio pronunciation and compares it to expected phonemes using Groq."""
    try:
        audio = request.files.get('audio')
        expected_text = request.form.get('expected_text')

        if not audio or not expected_text:
            return jsonify({"message": "Missing audio file or expected text!"}), 400

        # Save and preprocess audio
        audio_path = os.path.join(PROCESSED_DIR, "input_audio.wav")
        audio.save(audio_path)

        # Convert audio to 16kHz mono using ffmpeg
        processed_audio_path = os.path.join(PROCESSED_DIR, "processed_audio.wav")
        ffmpeg.input(audio_path).output(processed_audio_path, ar=16000, ac=1).run(overwrite_output=True)

        # Perform transcription with Groq
        with open(processed_audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(processed_audio_path, file.read()),
                model="whisper-large-v3-turbo",  # Replace with appropriate model
                prompt="Specify context or spelling",  # Optional
                response_format="json",  # Optional
                language="en",  # Optional
                temperature=0.0  # Optional
            )

        transcript_text = transcription['text'].strip()
        correctness = expected_text.strip().lower() in transcript_text.lower()

        return jsonify({
            "message": "Analysis complete",
            "transcript": transcript_text,
            "correct": correctness
        })

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/track_face', methods=['POST'])
def track_face():
    """Receives face movement tracking data from frontend."""
    try:
        data = request.json
        if not data or "movements" not in data:
            return jsonify({"message": "Invalid data!"}), 400

        movements = data["movements"]
        required_movements = {"left", "right", "up", "down"}
        passed = all(move in movements for move in required_movements)

        return jsonify({"message": "Face tracking analyzed", "valid_movements": passed})

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handles video uploads and processes lip-syncing with Gradio."""
    try:
        video = request.files.get('video')
        script = request.form.get('script')

        if not video or not script:
            return jsonify({"message": "Missing video or script!"}), 400

        video_file_path = os.path.join(PROCESSED_DIR, video.filename)
        video.save(video_file_path)

        # Generate TTS audio
        tts_audio_path = generate_tts_audio(script, os.path.join(PROCESSED_DIR, f'tts_{video.filename}.mp3'))

        if not os.path.exists(tts_audio_path):
            return jsonify({"message": "Failed to create TTS audio!"}), 500

        # Call Gradio Client API for lip-syncing
        result = client_gradio.predict(
            audio_path=file(tts_audio_path),
            video_path={"video": file(video_file_path)},
            bbox_shift=0,
            api_name="/inference"
        )

        output_video_url = result[0]["video"]

        # Save processed video to MongoDB
        with open(output_video_url, "rb") as f:
            final_video_id = fs.put(f, filename=f'final_{video.filename}', content_type="video/mp4")

        return jsonify({"message": "Video processed successfully!", "video_id": str(final_video_id)})

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route('/download/<video_id>', methods=['GET'])
def download_video(video_id):
    """Handles video downloads."""
    try:
        file_data = fs.get(ObjectId(video_id))
        return send_file(file_data, download_name=f"processed_video_{video_id}.mp4", as_attachment=True)

    except Exception as e:
        return jsonify({"message": f"Error downloading video: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

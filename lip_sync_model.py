import os
from flask import Flask, request, jsonify, send_file
from gradio_client import Client, file
from tts.generate_tts import generate_tts_audio
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId

app = Flask(__name__)

# MongoDB Setup
client = MongoClient("mongodb://root:OT9Xh66yfE3wkLuiTv59zpt1dI96zEgXTk2VQb8EHM1yPOUKuhu5tZq7PKbHy2hV@wc4cw8ck4ocskgk0oww08w0c:27017/?directConnection=true")  # Change accordingly
db = client["video_storage"]
fs = gridfs.GridFS(db)

# Local Directories for Processed Files
PROCESSED_DIR = 'static/processed_videos'
os.makedirs(PROCESSED_DIR, exist_ok=True)

client_gradio = Client("https://anhhayghen-musetalkv.hf.space/")

@app.route('/')
def index():
    return "Welcome to Lip Sync API"

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        video = request.files.get('video')
        script = request.form.get('script')

        if not video or not script:
            return jsonify({"message": "Missing video or script!"}), 400

        video_file_path = os.path.join(PROCESSED_DIR, video.filename)
        video.save(video_file_path)

        # Generate TTS audio from the script
        print("Generating TTS audio...")
        tts_audio_path = generate_tts_audio(script, os.path.join(PROCESSED_DIR, f'tts_{video.filename}.mp3'))

        if not os.path.exists(tts_audio_path):
            return jsonify({"message": "Failed to create TTS audio!"}), 500

        # Call Gradio Client API for lip-syncing
        print("Performing lip sync...")
        result = client_gradio.predict(
            audio_path=file(tts_audio_path),
            video_path={"video": file(video_file_path)},
            bbox_shift=0,
            api_name="/inference"
        )

        output_video_path = os.path.join(PROCESSED_DIR, f'final_{video.filename}')
        output_video_url = result[0]["video"]

        # Here you can download the processed video from the result or save it
        # Assuming it's returned as a URL or downloadable file

        # Save processed video to MongoDB (GridFS)
        with open(output_video_url, "rb") as f:
            final_video_id = fs.put(f, filename=f'final_{video.filename}', content_type="video/mp4")

        return jsonify({
            "message": "Video processed successfully!",
            "video_id": str(final_video_id)
        })

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route('/download/<video_id>', methods=['GET'])
def download_video(video_id):
    try:
        # Retrieve video from MongoDB GridFS
        video_file = fs.get(ObjectId(video_id))

        return send_file(
            video_file,
            mimetype=video_file.content_type,
            as_attachment=True,
            download_name=video_file.filename
        )

    except Exception as e:
        return jsonify({"message": "File not found!"}), 404

if __name__ == "__main__":
    app.run(debug=True)

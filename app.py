import os
import ffmpeg
import requests
from flask import Flask, request, jsonify, send_from_directory, send_file
from gradio_client import Client, file
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId

app = Flask(__name__)

# ---------------------------
# MongoDB & Local Storage Setup
# ---------------------------
client = MongoClient("mongodb://root:OT9Xh66yfE3wkLuiTv59zpt1dI96zEgXTk2VQb8EHM1yPOUKuhu5tZq7PKbHy2hV@wc4cw8ck4ocskgk0oww08w0c:27017/?directConnection=true")
db = client["video_storage"]
fs = gridfs.GridFS(db)

PROCESSED_DIR = 'static/processed_videos'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------------------------
# Gradio Client Setup
# ---------------------------
# Lip-sync API (unchangeable)
lip_sync_client = Client("https://anhhayghen-musetalkv.hf.space/")
# TTS API (for voice cloning)
tts_client = Client("https://nymbo-xtts-clone-voice-cpu.hf.space/--replicas/zyanz/")

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    # Serve the webpage (assumes index.html is in the "templates" folder)
    return send_from_directory('templates', 'index.html')

@app.route('/track_face', methods=['POST'])
def track_face():
    """
    Receives face tracking data from the frontend.
    (This simple endpoint checks that all required movement directions are present.)
    """
    try:
        data = request.json
        if not data or "movements" not in data:
            return jsonify({"message": "Invalid data!"}), 400

        movements = data["movements"]
        required_movements = {"left", "right", "up", "down"}
        valid = all(move in movements for move in required_movements)

        return jsonify({"message": "Face tracking analyzed", "valid_movements": valid})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Full lip-syncing pipeline:
      1. Save the uploaded sample video and custom script.
      2. Extract the audio from the sample video.
      3. Generate cloned TTS audio from the custom script using the extracted audio as reference.
      4. Call the (unchangeable) Gradio lip-sync API with the sample video and cloned audio.
      5. Post-process the returned video:
           a. Crop the video (to roughly isolate the lips).
           b. Apply a blur and overlay a face image.
           c. Combine (mux) the processed video with the cloned TTS audio.
      6. Save the final video to MongoDB (GridFS).
    """
    try:
        # Retrieve uploaded video and script, and optional toggle
        video = request.files.get('video')
        script = request.form.get('script')
        toggle_face_movement = request.form.get('toggle_face_movement', 'false').lower() == 'true'
        if not video or not script:
            return jsonify({"message": "Missing video or script!"}), 400

        # Save the sample video
        video_filename = video.filename
        video_file_path = os.path.join(PROCESSED_DIR, video_filename)
        video.save(video_file_path)

        # --- Step 1: Extract audio from the sample video ---
        extracted_audio_path = os.path.join(PROCESSED_DIR, "extracted_audio.wav")
        ffmpeg.input(video_file_path).output(extracted_audio_path, ac=1, ar=16000, format='wav').run(overwrite_output=True)

        # --- Step 2: Generate cloned TTS audio using the new Gradio API ---
        # Build a publicly accessible URL for the extracted audio.
        # (This uses the Flask host URL plus the relative path in the static folder.)
        reference_audio_url = f"{request.host_url}static/processed_videos/extracted_audio.wav"
        tts_result = tts_client.predict(
            script,               # Text prompt
            "en,en",              # Language
            reference_audio_url,  # Reference Audio (using extracted audio from sample video)
            reference_audio_url,  # Use Microphone for Reference (same extracted audio)
            True,                 # Check to use microphone as reference
            True,                 # Agree
            api_name="/predict"
        )
        # The TTS API returns a tuple; the synthesized audio is the second element.
        synth_audio_url = tts_result[1]
        # Download the synthesized audio locally
        tts_audio_path = os.path.join(PROCESSED_DIR, f'tts_{video_filename}.mp3')
        tts_response = requests.get(synth_audio_url)
        with open(tts_audio_path, "wb") as f:
            f.write(tts_response.content)

        if not os.path.exists(tts_audio_path):
            return jsonify({"message": "Failed to create TTS audio!"}), 500

        # --- Step 3: Call Gradio lip-sync API ---
        lip_sync_result = lip_sync_client.predict(
            audio_path=file(tts_audio_path),
            video_path={"video": file(video_file_path)},
            bbox_shift=0,
            api_name="/inference"
        )
        # The lip-sync API returns a tuple; we take the video file path from the first element.
        synced_video_url = lip_sync_result[0]["video"]

        # --- Step 4: Post-process the returned video ---
        # (a) Crop the video to isolate the lips.
        lips_video_path = os.path.join(PROCESSED_DIR, f'lips_{video_filename}')
        # Placeholder crop: full width and lower third of the height.
        ffmpeg.input(synced_video_url).filter('crop', 'iw', 'ih/3', 0, 'ih*2/3').output(lips_video_path).run(overwrite_output=True)

        # (b) Overlay a face image and apply a blur.
        # Make sure "static/face_overlay.png" exists.
        face_overlay = 'static/face_overlay.png'
        final_video_path = os.path.join(PROCESSED_DIR, f'final_{video_filename}')
        (
            ffmpeg
            .input(lips_video_path)
            .filter('boxblur', 10, 10)  # Apply a blur; adjust parameters as needed.
            .overlay(face_overlay, x='(main_w-overlay_w)/2', y='(main_h-overlay_h)/2')
            .output(final_video_path, an=None)  # Drop original audio.
            .run(overwrite_output=True)
        )

        # (c) Combine the processed (muted) video with the cloned TTS audio.
        output_video_path = os.path.join(PROCESSED_DIR, f'output_{video_filename}')
        ffmpeg.input(final_video_path).output(output_video_path, audio=tts_audio_path, vcodec='copy', acodec='aac').run(overwrite_output=True)

        # --- Step 5: Save the final video to MongoDB (GridFS) ---
        with open(output_video_path, "rb") as f:
            final_video_id = fs.put(f, filename=f'final_{video_filename}', content_type="video/mp4")

        return jsonify({"message": "Video processed successfully!", "video_id": str(final_video_id)})

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route('/download/<video_id>', methods=['GET'])
def download_video(video_id):
    """
    Retrieves the final processed video from MongoDB (GridFS) and sends it to the client.
    """
    try:
        video_file = fs.get(ObjectId(video_id))
        return send_file(
            video_file,
            mimetype=video_file.content_type,
            as_attachment=True,
            download_name=video_file.filename
        )
    except Exception as e:
        return jsonify({"message": f"Error downloading video: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

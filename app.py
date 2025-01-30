import os
import gridfs
from flask import Flask, request, jsonify, send_file, send_from_directory
from pymongo import MongoClient
from bson.objectid import ObjectId
from io import BytesIO
from lip_sync_model import LipSyncModel
from tts.generate_tts import generate_tts_audio
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

app = Flask(__name__)
lip_sync_model = LipSyncModel()

# MongoDB Setup
client = MongoClient("mongodb://root:OT9Xh66yfE3wkLuiTv59zpt1dI96zEgXTk2VQb8EHM1yPOUKuhu5tZq7PKbHy2hV@wc4cw8ck4ocskgk0oww08w0c:27017/?directConnection=true")  # Change if using MongoDB Atlas
db = client["video_storage"]
fs = gridfs.GridFS(db)

# Local Directories for Processed Files
PROCESSED_DIR = 'static/processed_videos'
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        video = request.files.get('video')
        script = request.form.get('script')

        if not video or not script:
            return jsonify({"message": "Missing video or script!"}), 400

        # Save video to MongoDB GridFS
        video_id = fs.put(video, filename=video.filename, content_type=video.content_type)

        print("Extracting audio...")
        audio_path = lip_sync_model.extract_audio_from_video(video)

        print("Generating phonemes...")
        phonemes = lip_sync_model.get_phonemes_from_script(script)

        print("Finding matching segments...")
        segments = lip_sync_model.find_matching_audio_segments(audio_path, phonemes)
        print(f"Segments found: {segments}")

        muted_video_path = os.path.join(PROCESSED_DIR, f"muted_{video.filename}")
        mute_video_segments(video, segments, muted_video_path)

        if not os.path.exists(muted_video_path):
            print("Muted video creation failed!")
            return jsonify({"message": "Failed to create muted video!"}), 500

        print("Generating TTS audio...")
        tts_audio_path = generate_tts_audio(script, os.path.join(PROCESSED_DIR, f'tts_{video.filename}.mp3'))

        if not os.path.exists(tts_audio_path):
            print("TTS generation failed!")
            return jsonify({"message": "Failed to create TTS audio!"}), 500

        final_video_path = os.path.join(PROCESSED_DIR, f'final_{video.filename}')
        print("Combining audio and video...")
        combine_audio_video(muted_video_path, tts_audio_path, final_video_path)

        if not os.path.exists(final_video_path):
            print("Final video creation failed!")
            return jsonify({"message": "Failed to create final video!"}), 500

        # Save processed video to MongoDB
        with open(final_video_path, "rb") as f:
            final_video_id = fs.put(f, filename=f'final_{video.filename}', content_type="video/mp4")

        return jsonify({
            "message": "Video processed successfully!",
            "video_id": str(final_video_id)
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

def mute_video_segments(video_path, segments, output_path):
    try:
        video = VideoFileClip(video_path)
        muted_clips = []
        last_end = 0

        for start, end in segments:
            if last_end < start:
                muted_clips.append(video.subclip(last_end, start).volumex(0))  # Mute unneeded parts
            muted_clips.append(video.subclip(start, end))
            last_end = end

        if last_end < video.duration:
            muted_clips.append(video.subclip(last_end, video.duration).volumex(0))

        final_video = concatenate_videoclips(muted_clips)
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)
    except Exception as e:
        print(f"Error muting video: {e}")
        raise

def combine_audio_video(video_path, audio_path, output_path):
    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        video = video.set_audio(audio)

        if audio.duration > video.duration:
            video = video.set_duration(audio.duration)

        video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
    except Exception as e:
        print(f"Error combining audio and video: {e}")
        raise

@app.route('/download/<video_id>', methods=['GET'])
def download_video(video_id):
    try:
        # Retrieve video from MongoDB GridFS
        video_file = fs.get(ObjectId(video_id))

        return send_file(
            BytesIO(video_file.read()),
            mimetype=video_file.content_type,
            as_attachment=True,
            download_name=video_file.filename
        )

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "File not found!"}), 404

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, send_from_directory
import os
from lip_sync_model import LipSyncModel
from tts.generate_tts import generate_tts_audio
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

app = Flask(__name__)
lip_sync_model = LipSyncModel()

# Ensure directories exist
UPLOAD_DIR = 'static/uploaded_videos'
PROCESSED_DIR = 'static/processed_videos'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        # Save uploaded video
        video = request.files['video']
        script = request.form['script']
        video_path = os.path.join(UPLOAD_DIR, video.filename)
        video.save(video_path)

        # Extract audio and process phonemes
        audio_path = lip_sync_model.extract_audio_from_video(video_path)
        phonemes = lip_sync_model.get_phonemes_from_script(script)
        segments = lip_sync_model.find_matching_audio_segments(audio_path, phonemes)

        if not segments:
            return jsonify({"message": "No matching audio segments found!"}), 400

        # Mute segments in the video
        muted_video_path = os.path.join(PROCESSED_DIR, 'muted_video.mp4')
        mute_video_segments(video_path, segments, muted_video_path)

        if not os.path.exists(muted_video_path):
            return jsonify({"message": "Failed to create muted video!"}), 500

        # Generate TTS audio
        tts_audio_path = generate_tts_audio(script, os.path.join(PROCESSED_DIR, 'tts_audio.mp3'))
        if not os.path.exists(tts_audio_path):
            return jsonify({"message": "Failed to create TTS audio!"}), 500

        # Combine audio and video
        final_video_path = os.path.join(PROCESSED_DIR, 'final_video.mp4')
        combine_audio_video(muted_video_path, tts_audio_path, final_video_path)

        if not os.path.exists(final_video_path):
            return jsonify({"message": "Failed to create final video!"}), 500

        return jsonify({"message": "Video processed successfully!", "video_path": final_video_path})

    except Exception as e:
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

@app.route('/download/<filename>', methods=['GET'])
def download_video(filename):
    try:
        file_path = os.path.join(PROCESSED_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({"message": "File not found!"}), 404

        response = send_from_directory(PROCESSED_DIR, filename, as_attachment=True)

        # Ensure the file is deleted after response is sent
        @response.call_on_close
        def remove_file():
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file: {e}")

        return response

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

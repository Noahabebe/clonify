import os
import subprocess
import numpy as np
import librosa
import ffmpeg
from flask import Flask, request, jsonify, send_from_directory, send_file
from pydub import AudioSegment
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
import nltk
from nltk.tokenize import word_tokenize
import mediapipe as mp
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Download CMU Dictionary if not available
nltk.download('cmudict', quiet=True)
nltk.download('punkt', quiet=True)
cmu_dict = nltk.corpus.cmudict.dict()

app = Flask(__name__)

# MongoDB & Local Storage Setup
client = MongoClient("mongodb://root:OT9Xh66yfE3wkLuiTv59zpt1dI96zEgXTk2VQb8EHM1yPOUKuhu5tZq7PKbHy2hV@wc4cw8ck4ocskgk0oww08w0c:27017/?directConnection=true")
db = client["video_storage"]
fs = gridfs.GridFS(db)

PROCESSED_DIR = 'static/processed_videos'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define the path to the models directory
MODELS_DIR = os.path.join(os.getcwd(), 'models')


# ---------------------------
# LipSyncModel Class
# ---------------------------
class LipSyncModel:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    def text_to_phonemes(self, word: str) -> list:
        """Convert a word into phonemes using CMU Dictionary with fallback."""
        word = word.lower()
        phoneme_list = cmu_dict.get(word, [])
        if phoneme_list and phoneme_list[0]:
            return phoneme_list[0]
        # Log a warning and use default fallback if not found.
        print(f"[WARNING] No phonemes found for word: '{word}'. Using default phoneme.")
        return ["AH0"]

    def extract_audio_from_video(self, video_path: str) -> str:
        """Extracts audio from a video file using ffmpeg and converts to WAV."""
        audio_path = video_path.replace('.mp4', '.wav')

        # Remove existing audio file to avoid conflicts
        if os.path.exists(audio_path):
            os.remove(audio_path)

        command = [
            "ffmpeg", "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "2", audio_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Debugging: Print FFmpeg output
        print("[DEBUG] FFmpeg Output:\n", result.stderr)

        # Check if file was created successfully and is nonempty
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            print("[ERROR] Audio extraction failed: File not created or empty.")
            return None

        print("[DEBUG] Audio extraction successful:", audio_path)
        return audio_path

    def get_phonemes_from_script(self, script: str) -> list:
        """Extract phonemes from the script with logging and validation."""
        phonemes = []
        # Use NLTK's word tokenizer to handle punctuation properly.
        words = word_tokenize(script)
        for word in words:
            word_phonemes = self.text_to_phonemes(word)
            print(f"[DEBUG] Word: {word}, Phonemes: {word_phonemes}")
            phonemes.extend(word_phonemes)
        return phonemes

    def find_matching_audio_segments(self, audio_path: str, phonemes: list) -> list:
        """Match phonemes to audio segments using MFCC features."""
        if not os.path.exists(audio_path):
            print(f"[ERROR] Audio file {audio_path} not found")
            return []

        segments = []
        y, sr = librosa.load(audio_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        for phoneme in phonemes:
            start, end = self.find_audio_segment_for_word(mfccs, phoneme)
            segments.append((start, end))

        return segments

    def find_audio_segment_for_word(self, mfccs: np.ndarray, phoneme: str) -> tuple:
        """Find time segment in audio matching phoneme using DTW."""
        word_features = self.phoneme_to_feature_matrix([phoneme])
        if word_features.shape[1] == 0 or mfccs.shape[1] == 0:
            return 0, 0  # Prevent out-of-range errors

        distance, path = fastdtw(word_features, mfccs.T, dist=euclidean)
        if len(path) == 0:
            return 0, 0

        start_idx, end_idx = path[0][0], path[-1][0]
        start_time = librosa.frames_to_time(start_idx, sr=22050)
        end_time = librosa.frames_to_time(end_idx, sr=22050)

        return start_time, end_time

    def phoneme_to_feature_matrix(self, phonemes: list) -> np.ndarray:
        """Convert phonemes to feature vectors with logging if a phoneme is missing."""
        phoneme_to_vector = {
            "AH0": [1, 0, 0], "EH0": [0, 1, 0], "IH0": [0, 0, 1],
            "AH1": [1, 1, 0], "EH1": [0, 1, 1], "IH1": [1, 0, 1],
            "AY1": [1, 1, 1]
        }
        features = []
        for phoneme in phonemes:
            vector = phoneme_to_vector.get(phoneme)
            if vector is None:
                print(f"[WARNING] No feature vector found for phoneme: {phoneme}. Using default [0, 0, 0].")
                vector = [0, 0, 0]
            features.append(vector)
        return np.array(features)


lip_sync_model = LipSyncModel()


# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/models/<path:filename>')
def serve_model(filename):
    """Serve model files from the 'models' directory."""
    return send_from_directory(MODELS_DIR, filename)
    
@app.after_request
def add_security_headers(response):
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    return response


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        video = request.files.get('video')
        script = request.form.get('script')

        # Validate that both video and script are provided.
        if not video:
            return jsonify({"message": "Missing video file!"}), 400
        if not script or script.strip() == "":
            return jsonify({"message": "Script cannot be empty!"}), 400

        # Validate video file extension.
        if not video.filename.lower().endswith(".mp4"):
            return jsonify({"message": "Invalid video file format. Only MP4 allowed."}), 400

        video_filename = video.filename
        video_file_path = os.path.join(PROCESSED_DIR, video_filename)
        video.save(video_file_path)

        extracted_audio_path = lip_sync_model.extract_audio_from_video(video_file_path)
        if not extracted_audio_path:
            return jsonify({"message": "Audio extraction failed!"}), 500

        phonemes = lip_sync_model.get_phonemes_from_script(script)
        audio_segments = lip_sync_model.find_matching_audio_segments(extracted_audio_path, phonemes)

        synced_video_path = apply_lip_sync_to_video(video_file_path, extracted_audio_path, audio_segments)

        with open(synced_video_path, "rb") as f:
            final_video_id = fs.put(f, filename=f'final_{video_filename}', content_type="video/mp4")

        return jsonify({"message": "Video processed successfully!", "video_id": str(final_video_id)})

    except Exception as e:
        print(f"[ERROR] Exception in /upload: {e}")
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500


def apply_lip_sync_to_video(video_path: str, audio_path: str, audio_segments: list) -> str:
    """
    Apply lip sync to the video with the processed audio.

    Placeholder implementation:
      - Extracts segments from the original audio corresponding to detected phoneme timings.
      - Concatenates these segments into a new audio file.
      - Overlays the new audio track on the video.
    Note: A proper lip sync would adjust video frames based on phoneme timing.
    """
    new_audio_path = video_path.replace('.mp4', '_new_audio.wav')
    if os.path.exists(new_audio_path):
        os.remove(new_audio_path)

    try:
        # Load original audio with pydub
        original_audio = AudioSegment.from_wav(audio_path)
        new_audio = AudioSegment.empty()

        # Process each detected audio segment.
        for (start, end) in audio_segments:
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            if end_ms > start_ms:
                segment_audio = original_audio[start_ms:end_ms]
                new_audio += segment_audio

        # If no valid segments were found, fall back to the original audio.
        if len(new_audio) == 0:
            new_audio = original_audio

        new_audio.export(new_audio_path, format="wav")
    except Exception as e:
        print(f"[ERROR] Failed to create new audio track: {e}")
        new_audio_path = audio_path

    # Merge the new audio track with the video.
    synced_video_path = video_path.replace(".mp4", "_synced.mp4")
    try:
        ffmpeg.input(video_path).output(synced_video_path, audio=new_audio_path).run(overwrite_output=True)
        print(f"[DEBUG] Synced video created: {synced_video_path}")
    except Exception as e:
        print(f"[ERROR] Lip sync video processing failed: {e}")
        return video_path  # Fallback to original video if processing fails.
    return synced_video_path


@app.route('/download/<video_id>', methods=['GET'])
def download_video(video_id):
    try:
        video_file = fs.get(ObjectId(video_id))
        return send_file(video_file, mimetype=video_file.content_type, as_attachment=True, download_name=video_file.filename)
    except Exception as e:
        print(f"[ERROR] Exception in /download: {e}")
        return jsonify({"message": f"Error downloading video: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)

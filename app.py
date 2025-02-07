import os
import requests
import numpy as np
import librosa
import ffmpeg
from flask import Flask, request, jsonify, send_from_directory
from pydub import AudioSegment
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
from nltk.corpus import cmudict
import nltk
import mediapipe as mp

nltk.download('cmudict', force=True)
cmu_dict = cmudict.dict()

app = Flask(__name__)

# MongoDB & Local Storage Setup
client = MongoClient("mongodb://root:OT9Xh66yfE3wkLuiTv59zpt1dI96zEgXTk2VQb8EHM1yPOUKuhu5tZq7PKbHy2hV@wc4cw8ck4ocskgk0oww08w0c:27017/?directConnection=true")
db = client["video_storage"]
fs = gridfs.GridFS(db)

PROCESSED_DIR = 'static/processed_videos'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# LipSyncModel Class as per your custom code
class LipSyncModel:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    def text_to_phonemes(self, word: str) -> list:
        word = word.lower()
        return cmu_dict.get(word, [[]])[0]

    def extract_audio_from_video(self, video_path: str) -> str:
        audio_path = video_path.replace('.mp4', '.wav')
        audio = AudioSegment.from_file(video_path, format="mp4")
        audio.export(audio_path, format="wav")
        return audio_path

    def get_phonemes_from_script(self, script: str) -> list:
        phonemes = []
        for word in script.split():
            phonemes.extend(self.text_to_phonemes(word))
        return phonemes

    def find_matching_audio_segments(self, audio_path: str, phonemes: list) -> list:
        segments = []
        y, sr = librosa.load(audio_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for phoneme in phonemes:
            start, end = self.find_audio_segment_for_word(mfccs, phoneme)
            segments.append((start, end))
        return segments

    def find_audio_segment_for_word(self, mfccs: np.ndarray, phoneme: str) -> tuple:
        word_features = self.phoneme_to_feature_matrix([phoneme])
        if word_features.shape[1] != mfccs.T.shape[1]:
            return 0, 0
        distance, path = fastdtw(word_features, mfccs.T, dist=euclidean)
        start_idx, end_idx = path[0][0], path[-1][0]
        start_time = librosa.frames_to_time(start_idx, sr=22050)
        end_time = librosa.frames_to_time(end_idx, sr=22050)
        return start_time, end_time

    def phoneme_to_feature_matrix(self, phonemes: list) -> np.ndarray:
        phoneme_to_vector = {
            "AH0": [1, 0, 0], "EH0": [0, 1, 0], "IH0": [0, 0, 1],
            "AH1": [1, 1, 0], "EH1": [0, 1, 1], "IH1": [1, 0, 1], "AY1": [1, 1, 1]
        }
        return np.array([phoneme_to_vector.get(phoneme, [0, 0, 0]) for phoneme in phonemes])

lip_sync_model = LipSyncModel()

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

# Define the path to the models directory
MODELS_DIR = os.path.join(os.getcwd(), 'models')

@app.route('/models/<path:filename>')
def serve_model(filename):
    """Serve model files from the 'models' directory."""
    return send_from_directory(MODELS_DIR, filename)
    

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        video = request.files.get('video')
        script = request.form.get('script')
        if not video or not script:
            return jsonify({"message": "Missing video or script!"}), 400

        video_filename = video.filename
        video_file_path = os.path.join(PROCESSED_DIR, video_filename)
        video.save(video_file_path)

        # Step 1: Extract audio from video
        extracted_audio_path = lip_sync_model.extract_audio_from_video(video_file_path)

        # Step 2: Generate phonemes from the script
        phonemes = lip_sync_model.get_phonemes_from_script(script)

        # Step 3: Match audio segments with phonemes
        audio_segments = lip_sync_model.find_matching_audio_segments(extracted_audio_path, phonemes)

        # Step 4: Perform voice cloning (OpenVoice integration)
        # Assume voice cloning is done via OpenVoice API or local setup and the synthesized audio is returned
        cloned_audio_path = generate_voice_cloning_audio(script)

        # Step 5: Apply lip sync to the video with the generated audio and face tracking
        synced_video_path = apply_lip_sync_to_video(video_file_path, cloned_audio_path, audio_segments)

        # Step 6: Save final video to MongoDB (GridFS)
        with open(synced_video_path, "rb") as f:
            final_video_id = fs.put(f, filename=f'final_{video_filename}', content_type="video/mp4")

        return jsonify({"message": "Video processed successfully!", "video_id": str(final_video_id)})

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

def generate_voice_cloning_audio(script: str, video_path: str) -> str:
    try:
        # Step 1: Extract audio from the provided video file
        lip_sync_model = LipSyncModel()
        extracted_audio_path = lip_sync_model.extract_audio_from_video(video_path)
        
        # Step 2: Select the language for voice cloning (you can replace 'en' with any supported language)
        language = 'en'  # Example language: English
        
        # Step 3: Initialize the client to interact with the OpenVoice model API
        client = Client("luigi12345/Voice-Clone-Multilingual")

        # Step 4: Call the API to generate the voice-cloned audio from the script and extracted audio
        result = client.predict(
            text=script,  # The script to be cloned
            speaker_wav=handle_file(extracted_audio_path),  # Path to the extracted audio file
            language=language,  # Language of the script
            api_name="/predict"  # API endpoint for voice cloning
        )
        
        # Step 5: The API response contains a filepath to the generated audio file
        cloned_audio_path = result[0]
        
        # Step 6: Save the audio file locally
        cloned_audio_local_path = "static/cloned_audio.wav"
        response = requests.get(cloned_audio_path, stream=True)
        with open(cloned_audio_local_path, 'wb') as out_file:
            out_file.write(response.content)
        
        return cloned_audio_local_path
    
    except Exception as e:
        print(f"Error generating voice cloning audio: {str(e)}")
        return None

# Lip Sync function to apply lip sync to video
def apply_lip_sync_to_video(video_path: str, audio_path: str, audio_segments: list) -> str:
    # Lip sync logic based on face tracking and the audio segments
    synced_video_path = video_path.replace(".mp4", "_synced.mp4")
    # Process video with audio and lip sync
    ffmpeg.input(video_path).output(synced_video_path, audio=audio_path).run(overwrite_output=True)
    return synced_video_path

@app.route('/download/<video_id>', methods=['GET'])
def download_video(video_id):
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

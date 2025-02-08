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
import cv2
from rembg import remove
import tempfile

# Download required NLTK resources if needed
nltk.download('cmudict', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
cmu_dict = nltk.corpus.cmudict.dict()

app = Flask(__name__)

# MongoDB & Local Storage Setup
client = MongoClient("mongodb://root:password@localhost:27017/?directConnection=true")
db = client["video_storage"]
fs = gridfs.GridFS(db)
PROCESSED_DIR = 'static/processed_videos'
os.makedirs(PROCESSED_DIR, exist_ok=True)
MODELS_DIR = os.path.join(os.getcwd(), 'models')

# ---------------------------
# LipSyncModel Class (existing functionality)
# ---------------------------
class LipSyncModel:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    def text_to_phonemes(self, word: str) -> list:
        word = word.lower()
        phoneme_list = cmu_dict.get(word, [])
        if phoneme_list and phoneme_list[0]:
            return phoneme_list[0]
        print(f"[WARNING] No phonemes found for word: '{word}'. Using default phoneme.")
        return ["AH0"]

    def extract_audio_from_video(self, video_path: str) -> str:
        base, _ = os.path.splitext(video_path)
        audio_path = base + '.wav'
        if os.path.exists(audio_path):
            os.remove(audio_path)
        command = [
            "ffmpeg", "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "2", audio_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("[DEBUG] FFmpeg Output:\n", result.stderr)
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            print("[ERROR] Audio extraction failed: File not created or empty.")
            return None
        print("[DEBUG] Audio extraction successful:", audio_path)
        return audio_path

    def get_phonemes_from_script(self, script: str) -> list:
        phonemes = []
        words = word_tokenize(script)
        for word in words:
            word_phonemes = self.text_to_phonemes(word)
            print(f"[DEBUG] Word: {word}, Phonemes: {word_phonemes}")
            phonemes.extend(word_phonemes)
        return phonemes

    def find_matching_audio_segments(self, audio_path: str, phonemes: list) -> list:
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
        word_features = self.phoneme_to_feature_matrix([phoneme])
        if word_features.shape[1] == 0 or mfccs.shape[1] == 0:
            return 0, 0
        distance, path = fastdtw(word_features, mfccs.T, dist=euclidean)
        if len(path) == 0:
            return 0, 0
        start_idx, end_idx = path[0][0], path[-1][0]
        start_time = librosa.frames_to_time(start_idx, sr=22050)
        end_time = librosa.frames_to_time(end_idx, sr=22050)
        return start_time, end_time

    def phoneme_to_feature_matrix(self, phonemes: list) -> np.ndarray:
        phoneme_to_vector = {
            "AH0": [1, 0, 0],
            "EH0": [0, 1, 0],
            "IH0": [0, 0, 1],
            "AH1": [1, 1, 0],
            "EH1": [0, 1, 1],
            "IH1": [1, 0, 1],
            "AY1": [1, 1, 1]
        }
        features = []
        for phoneme in phonemes:
            vector = phoneme_to_vector.get(phoneme)
            if vector is None:
                print(f"[WARNING] No feature vector found for phoneme: {phoneme}. Using default [0, 0, 0].")
                vector = [0, 0, 0]
            # Pad vector to 13 dimensions
            padded_vector = vector + [0]*(13 - len(vector))
            features.append(padded_vector)
        return np.array(features)

lip_sync_model = LipSyncModel()

# ---------------------------
# AI and Video Processing Functions
# ---------------------------

def apply_ai_lip_sync(video_path: str, audio_path: str) -> str:
    """
    Use an AI model (e.g., Wav2Lip) to generate a lip-synced video.
    This function assumes you have a Wav2Lip inference script available.
    """
    inference_script = os.path.abspath('Wav2Lip/inference.py')
    checkpoint_path = os.path.abspath('Wav2Lip/checkpoint/wav2lip.pth')
    output_video = os.path.splitext(video_path)[0] + '_ai_lipsynced.mp4'

    try:
        command = [
            'python', inference_script,
            '--checkpoint_path', checkpoint_path,
            '--face', os.path.abspath(video_path),
            '--audio', os.path.abspath(audio_path),
            '--outfile', output_video
        ]
        subprocess.run(command, check=True)
        print(f"AI Lip Sync complete: {output_video}")
    except Exception as e:
        print(f"AI Lip Sync failed: {e}")
        return video_path

    return output_video

def apply_face_movement(video_path: str) -> str:
    """
    Simulate face movement by detecting facial landmarks (via Mediapipe)
    and applying a slight warp or translation to the face region.
    """
    output_path = os.path.splitext(video_path)[0] + '_face_moved.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video for face movement processing.")
        return video_path

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            # For demonstration, we shift the face slightly to the right
            M = np.float32([[1, 0, 5], [0, 1, 0]])
            frame = cv2.warpAffine(frame, M, (width, height))
        out.write(frame)
    cap.release()
    out.release()
    print(f"[DEBUG] Face movement applied: {output_path}")
    return output_path

def apply_face_blur(video_path: str) -> str:
    """
    Apply a blur effect to the face region.
    Uses a simple Haar Cascade for face detection.
    """
    output_path = os.path.splitext(video_path)[0] + '_face_blurred.mp4'
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video for face blur processing.")
        return video_path

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            # Blur the face region
            face_region = frame[y:y+h, x:x+w]
            face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y:y+h, x:x+w] = face_region
        out.write(frame)
    cap.release()
    out.release()
    print(f"[DEBUG] Face blur applied: {output_path}")
    return output_path

def apply_background_removal(video_path: str) -> str:
    """
    Remove the background from each frame using the 'rembg' package.
    This will output a video with a transparent or plain background.
    """
    output_path = os.path.splitext(video_path)[0] + '_bg_removed.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video for background removal processing.")
        return video_path

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to PNG bytes and remove background
        _, buffer = cv2.imencode('.png', frame)
        result = remove(buffer.tobytes())
        # Decode the processed image
        nparr = np.frombuffer(result, np.uint8)
        processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if processed_frame is None:
            processed_frame = frame
        else:
            processed_frame = cv2.resize(processed_frame, (width, height))
        out.write(processed_frame)
    cap.release()
    out.release()
    print(f"[DEBUG] Background removal applied: {output_path}")
    return output_path

def process_video_advanced(video_path: str, audio_path: str, toggles: dict) -> str:
    """
    Process the video based on toggles:
      - AI Lip Sync
      - Face Movement
      - Face Blur
      - Background Removal
    Each step creates a new video file (using suffixes) and the output of one step is fed into the next.
    """
    current_video = video_path

    if toggles.get('ai_lip_sync'):
        current_video = apply_ai_lip_sync(current_video, audio_path)

    if toggles.get('face_movement'):
        current_video = apply_face_movement(current_video)

    if toggles.get('blur'):
        current_video = apply_face_blur(current_video)

    if toggles.get('background_removal'):
        current_video = apply_background_removal(current_video)

    return current_video

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/models/<path:filename>')
def serve_model(filename):
    return send_from_directory(MODELS_DIR, filename)

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        video = request.files.get('video')
        script = request.form.get('script')

        if not video:
            return jsonify({"message": "Missing video file!"}), 400
        if not script or script.strip() == "":
            return jsonify({"message": "Script cannot be empty!"}), 400

        # Allow MP4 and WEBM uploads
        if not (video.filename.lower().endswith(".mp4") or video.filename.lower().endswith(".webm")):
            return jsonify({"message": "Invalid video file format. Only MP4 or WEBM allowed."}), 400

        video_filename = video.filename
        video_file_path = os.path.join(PROCESSED_DIR, video_filename)
        video.save(video_file_path)

        # Extract audio from video
        extracted_audio_path = lip_sync_model.extract_audio_from_video(video_file_path)
        if not extracted_audio_path:
            return jsonify({"message": "Audio extraction failed!"}), 500

        # (Optional) Use phoneme matching if desired – here we simply log segments.
        phonemes = lip_sync_model.get_phonemes_from_script(script)
        audio_segments = lip_sync_model.find_matching_audio_segments(extracted_audio_path, phonemes)
        for i, (start, end) in enumerate(audio_segments):
            print(f"[DEBUG] Audio segment {i}: {start:.2f} s to {end:.2f} s")

        # Get toggle values from the form (checkbox value "on" if enabled)
        toggles = {
            'face_movement': request.form.get('toggle_face_movement') == "on",
            'blur': request.form.get('toggle_blur') == "on",
            'background_removal': request.form.get('toggle_background_removal') == "on",
            'ai_lip_sync': request.form.get('toggle_ai_lip_sync') == "on"
        }

        # Process the video using advanced AI methods and other effects
        processed_video_path = process_video_advanced(video_file_path, extracted_audio_path, toggles)

        # (Optionally, merge the processed audio back if needed; here we assume the AI lip sync already handled audio.)
        # For demonstration, we assume the processed video already has proper audio.
        with open(processed_video_path, "rb") as f:
            final_video_id = fs.put(f, filename=f'final_{video_filename}', content_type="video/mp4")

        return jsonify({"message": "Video processed successfully!", "video_id": str(final_video_id)})

    except Exception as e:
        print(f"[ERROR] Exception in /upload: {e}")
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

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

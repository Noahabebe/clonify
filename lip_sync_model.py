import librosa
import numpy as np
from nltk.corpus import cmudict
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import mediapipe as mp
from pydub import AudioSegment
import nltk
nltk.download('cmudict')

cmu_dict = cmudict.dict()

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

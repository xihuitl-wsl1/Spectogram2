import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import librosa.display


class SpectrogramComparator:
    def __init__(self, file1: str, file2: str):
        self.file1 = file1
        self.file2 = file2
        self.sr1 = None
        self.sr2 = None
        self.S1 = None
        self.S2 = None

    def _load_audio(self):
        y1, self.sr1 = librosa.load(self.file1)
        y2, self.sr2 = librosa.load(self.file2)
        return y1, y2

    def _compute_spectrograms(self, y1, y2):
        S1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
        S2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)
        return S1, S2

    def compare(self) -> float:
        # Load audio
        y1, y2 = self._load_audio()

        # Compute spectrograms
        self.S1, self.S2 = self._compute_spectrograms(y1, y2)

        # Resize both spectrograms to the same shape
        min_shape = (
            min(self.S1.shape[0], self.S2.shape[0]),
            min(self.S1.shape[1], self.S2.shape[1])
        )
        S1_resized = self.S1[:min_shape[0], :min_shape[1]]
        S2_resized = self.S2[:min_shape[0], :min_shape[1]]

        # Flatten for similarity calculation
        S1_flat = S1_resized.flatten().reshape(1, -1)
        S2_flat = S2_resized.flatten().reshape(1, -1)

        # Cosine similarity (range -1 to 1)
        similarity = cosine_similarity(S1_flat, S2_flat)[0, 0]

        # Normalize to percentage 0–100%
        similarity_percent = (similarity + 1) / 2 * 100
        return similarity_percent


class SpectrogramAnalysis:
    def __init__(self, file1: str, file2: str):
        self.comparator = SpectrogramComparator(file1, file2)

    def get_similarity(self) -> float:
        return self.comparator.compare()

    def plot_spectrograms(self):
        if self.comparator.S1 is None or self.comparator.S2 is None:
            # Make sure spectrograms are computed
            self.comparator.compare()

        S1, S2 = self.comparator.S1, self.comparator.S2
        sr1, sr2 = self.comparator.sr1, self.comparator.sr2
        similarity = self.get_similarity()

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        librosa.display.specshow(S1, sr=sr1, x_axis='time', y_axis='hz', cmap='magma')
        plt.title("Spectrogram: File 1")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        librosa.display.specshow(S2, sr=sr2, x_axis='time', y_axis='hz', cmap='magma')
        plt.title("Spectrogram: File 2")
        plt.colorbar()

        plt.suptitle(f"Similarity: {similarity:.2f}%", fontsize=14, y=1.05)
        plt.tight_layout()
        plt.show()

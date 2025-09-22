from SpectogramComparator import SpectrogramAnalysis

analysis = SpectrogramAnalysis("ClipsSpecto/Alana-Conv.wav", "ClipsSpecto/Alana-Narration.wav")

# Get similarity as percentage
similarity = analysis.get_similarity()
print(f"Spectrogram similarity: {similarity:.2f}%")

# Plot spectrograms with similarity
analysis.plot_spectrograms()

import numpy as np
import mne
import pandas as pd
from scipy import signal

class EEGPreprocessor:
    def __init__(self, sampling_rate=250, channels=64):
        """
        Initialize EEG Preprocessor
        
        Args:
            sampling_rate (int): EEG signal sampling rate
            channels (int): Number of EEG channels
        """
        self.sampling_rate = sampling_rate
        self.channels = channels
    
    def load_raw_data(self, file_path):
        """
        Load raw EEG data from file
        
        Args:
            file_path (str): Path to EEG data file
        
        Returns:
            mne.io.Raw: Raw EEG data
        """
        try:
            raw = mne.io.read_raw_bdf(file_path, preload=True)
            return raw
        except Exception as e:
            print(f"Error loading EEG data: {e}")
            return None
    
    def preprocess_signal(self, raw):
        """
        Comprehensive EEG signal preprocessing
        
        Args:
            raw (mne.io.Raw): Raw EEG data
        
        Returns:
            np.ndarray: Preprocessed signal
        """
        # 1. Filtering
        raw.filter(l_freq=1, h_freq=50)  # Bandpass filter
        
        # 2. Remove power line noise (50/60 Hz)
        raw.notch_filter(freqs=[50, 60])
        
        # 3. Artifact removal (Basic ICA)
        ica = mne.preprocessing.ICA(n_components=15, random_state=42)
        ica.fit(raw)
        raw_clean = ica.apply(raw)
        
        # 4. Extract data as numpy array
        data = raw_clean.get_data()
        
        return data
    
    def extract_motor_imagery_features(self, preprocessed_data):
        """
        Extract features specific to motor imagery
        
        Args:
            preprocessed_data (np.ndarray): Preprocessed EEG signal
        
        Returns:
            np.ndarray: Motor imagery features
        """
        # Compute power spectral density
        psds, freqs = mne.time_frequency.psd_welch(preprocessed_data)
        
        # Focus on motor imagery relevant frequency bands
        motor_bands = {
            'mu': (8, 12),     # Mu rhythm
            'beta': (13, 30)   # Beta rhythm
        }
        
        features = []
        for band, (low, high) in motor_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.mean(psds[:, band_mask], axis=1)
            features.append(band_power)
        
        return np.array(features).T
    
    def segment_signals(self, data, window_size=1, overlap=0.5):
        """
        Segment EEG signals into fixed-length windows
        
        Args:
            data (np.ndarray): Preprocessed EEG data
            window_size (float): Window size in seconds
            overlap (float): Overlap between windows
        
        Returns:
            list: Segmented signal windows
        """
        window_samples = int(window_size * self.sampling_rate)
        overlap_samples = int(window_samples * overlap)
        
        segments = []
        for i in range(0, data.shape[1] - window_samples, window_samples - overlap_samples):
            segment = data[:, i:i+window_samples]
            segments.append(segment)
        
        return np.array(segments)

# Example usage
def main():
    preprocessor = EEGPreprocessor()
    
    # Load your EEG data file
    raw_data = preprocessor.load_raw_data(r'D:/Major project/eeg_speech/EEG_to_speechno/MindBigDataVisualMnist2021-Muse2v0.16Cut2.csv')
    
    if raw_data is not None:
        # Preprocess the signal
        preprocessed_signal = preprocessor.preprocess_signal(raw_data)
        
        # Extract motor imagery features
        features = preprocessor.extract_motor_imagery_features(preprocessed_signal)
        
        # Segment signals
        segmented_signals = preprocessor.segment_signals(preprocessed_signal)
        
        print("Preprocessing Complete!")
        print(f"Features shape: {features.shape}")
        print(f"Segmented signals shape: {segmented_signals.shape}")

if __name__ == "__main__":
    main()


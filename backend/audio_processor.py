"""
Audio Processor for Brein AI - Multi-modal Speech and Audio Analysis
Implements speech recognition, audio analysis, and sound pattern processing capabilities.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import io
import wave
import sys
import warnings
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

# Suppress FFmpeg warning from pydub
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", category=RuntimeWarning)

# Try to import audio processing dependencies
try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    import speech_recognition as sr
    AUDIO_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Audio processing dependencies not available: {e}")
    logger.info("Audio processing will be disabled")
    AUDIO_DEPENDENCIES_AVAILABLE = False
    AudioSegment = None
    detect_nonsilent = None
    sr = None

class AudioProcessor:
    """
    Advanced audio processing for speech recognition, audio analysis, and sound understanding.
    Handles multiple audio formats and provides comprehensive audio feature extraction.
    """

    def __init__(self, cache_dir: str = "memory/audio_cache"):
        """
        Initialize the audio processor.

        Args:
            cache_dir: Directory to cache audio processing results
        """
        self.available = AUDIO_DEPENDENCIES_AVAILABLE
        self.cache_dir = cache_dir

        if not self.available:
            logger.warning("Audio processing dependencies not available. Audio processor disabled.")
            # Initialize minimal attributes for disabled state
            self.audio_cache = {}
            self.cache_max_size = 0
            return

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize speech recognition
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300  # Adjust for ambient noise
            self.recognizer.dynamic_energy_threshold = True
        except Exception as e:
            logger.error(f"Failed to initialize speech recognizer: {e}")
            self.available = False
            self.audio_cache = {}
            self.cache_max_size = 0
            return

        # Audio processing cache
        self.audio_cache: Dict[str, Dict] = {}
        self.cache_max_size = 200

        # Audio analysis parameters
        self.sample_rate = 16000
        self.chunk_size = 1024

        # Supported audio formats
        self.supported_formats = ['wav', 'mp3', 'flac', 'ogg', 'm4a']

    def process_audio(self, audio_input: Any, context: Optional[Dict] = None,
                     memory_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process audio input and extract speech, features, and semantic information.

        Args:
            audio_input: Audio file path, bytes, or AudioSegment
            context: Optional context information
            memory_id: Optional memory ID to link concepts to

        Returns:
            Dictionary containing audio analysis results
        """
        # Load and preprocess audio
        audio_segment = self._load_audio(audio_input)
        if audio_segment is None:
            return {"error": "Failed to load audio"}

        # Generate audio signature for caching
        audio_signature = self._generate_audio_signature(audio_segment)

        # Check cache first
        if audio_signature in self.audio_cache:
            cached_result = self.audio_cache[audio_signature].copy()
            cached_result['cached'] = True
            return cached_result

        # Process audio
        try:
            # Extract audio features
            audio_features = self._extract_audio_features(audio_segment)

            # Perform speech recognition
            transcription = self._transcribe_speech(audio_segment)

            # Analyze audio content
            content_analysis = self._analyze_audio_content(audio_segment)

            # Extract audio concepts
            concepts = self._extract_audio_concepts(audio_segment, transcription, context)

            # Detect emotions and sentiment
            emotion_analysis = self._analyze_emotion_and_sentiment(audio_segment, transcription)

            # Create result
            result = {
                'audio_signature': audio_signature,
                'transcription': transcription,
                'audio_features': audio_features,
                'content_analysis': content_analysis,
                'concepts': concepts,
                'emotion_analysis': emotion_analysis,
                'processing_timestamp': datetime.now().isoformat(),
                'context': context,
                'memory_id': memory_id,
                'cached': False
            }

            # Cache result
            self._cache_result(audio_signature, result)

            return result

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {"error": f"Audio processing failed: {str(e)}"}

    def _load_audio(self, audio_input: Any) -> Optional[AudioSegment]:
        """Load audio from various input formats."""
        try:
            if isinstance(audio_input, AudioSegment):
                return audio_input
            elif isinstance(audio_input, str):
                if os.path.exists(audio_input):
                    # File path
                    file_ext = os.path.splitext(audio_input)[1].lower()
                    if file_ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                        return AudioSegment.from_file(audio_input)
                    else:
                        return None
                else:
                    # Assume base64 or raw data - not implemented yet
                    return None
            elif isinstance(audio_input, bytes):
                # Try to load as WAV
                try:
                    return AudioSegment.from_wav(io.BytesIO(audio_input))
                except:
                    # Try other formats
                    try:
                        return AudioSegment.from_mp3(io.BytesIO(audio_input))
                    except:
                        return None
            elif isinstance(audio_input, sr.AudioData):
                # Convert from speech recognition AudioData
                wav_data = io.BytesIO()
                wav_file = wave.open(wav_data, 'wb')
                wav_file.setnchannels(audio_input.channels)
                wav_file.setsampwidth(audio_input.sample_width)
                wav_file.setframerate(audio_input.sample_rate)
                wav_file.writeframes(audio_input.frame_data)
                wav_file.close()
                wav_data.seek(0)
                return AudioSegment.from_wav(wav_data)
            else:
                return None
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return None

    def _generate_audio_signature(self, audio: AudioSegment) -> str:
        """Generate a unique signature for audio caching."""
        # Convert to raw audio data for hashing
        raw_audio = audio.raw_data[:min(len(audio.raw_data), 44100)]  # First second

        import hashlib
        return hashlib.md5(raw_audio).hexdigest()[:16]

    def _extract_audio_features(self, audio: AudioSegment) -> Dict[str, Any]:
        """Extract low-level audio features."""
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            # Use mono for analysis
            samples = samples.mean(axis=1)

        # Basic audio statistics
        duration = len(audio) / 1000.0  # Duration in seconds
        rms = np.sqrt(np.mean(samples**2))
        peak = np.max(np.abs(samples))

        # Zero crossing rate (rough voice activity detection)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(samples)))) / (2 * len(samples))

        # Spectral centroid approximation (simplified)
        # This is a very basic approximation - real spectral analysis would use FFT
        spectral_centroid = np.mean(np.abs(samples))

        return {
            'duration_seconds': duration,
            'sample_rate': audio.frame_rate,
            'channels': audio.channels,
            'rms_amplitude': float(rms),
            'peak_amplitude': float(peak),
            'zero_crossing_rate': float(zero_crossings),
            'estimated_spectral_centroid': float(spectral_centroid),
            'bit_depth': audio.sample_width * 8
        }

    def _transcribe_speech(self, audio: AudioSegment) -> Dict[str, Any]:
        """Transcribe speech from audio using speech recognition."""
        transcription_result = {
            'text': '',
            'confidence': 0.0,
            'language': 'en-US',
            'segments': [],
            'success': False
        }

        try:
            # Convert AudioSegment to format suitable for speech recognition
            wav_data = io.BytesIO()
            audio.export(wav_data, format='wav')
            wav_data.seek(0)

            # Create AudioFile for speech recognition
            with sr.AudioFile(wav_data) as source:
                audio_data = self.recognizer.record(source)

                # Perform speech recognition
                try:
                    text = self.recognizer.recognize_google(audio_data)
                    transcription_result.update({
                        'text': text,
                        'confidence': 0.8,  # Google doesn't provide confidence
                        'success': True
                    })
                except sr.UnknownValueError:
                    transcription_result['text'] = "[Speech not recognized]"
                except sr.RequestError as e:
                    transcription_result['text'] = f"[Speech recognition error: {e}]"

        except Exception as e:
            logger.warning(f"Speech transcription failed: {e}")
            transcription_result['text'] = f"[Transcription failed: {str(e)}]"

        return transcription_result

    def _analyze_audio_content(self, audio: AudioSegment) -> Dict[str, Any]:
        """Analyze audio content for patterns and characteristics."""
        # Detect silence and speech segments
        nonsilent_chunks = detect_nonsilent(
            audio,
            min_silence_len=500,  # 500ms
            silence_thresh=-40    # dBFS
        )

        # Calculate speech percentage
        total_duration = len(audio)
        speech_duration = sum(end - start for start, end in nonsilent_chunks)
        speech_percentage = (speech_duration / total_duration) * 100 if total_duration > 0 else 0

        # Basic audio quality assessment
        rms = audio.rms
        quality_score = min(1.0, rms / 10000)  # Normalize RMS to quality score

        # Detect audio type (music, speech, noise)
        audio_type = self._classify_audio_type(audio, nonsilent_chunks)

        return {
            'speech_percentage': speech_percentage,
            'num_speech_segments': len(nonsilent_chunks),
            'audio_quality_score': quality_score,
            'detected_audio_type': audio_type,
            'speech_segments': nonsilent_chunks[:10]  # First 10 segments
        }

    def _classify_audio_type(self, audio: AudioSegment, speech_segments: List) -> str:
        """Classify audio as speech, music, or noise."""
        # Simple heuristic-based classification
        speech_percentage = len(speech_segments) * 100 / max(len(audio) / 1000, 1)  # Per second

        if speech_percentage > 50:
            return "speech"
        elif speech_percentage > 20:
            return "mixed"
        else:
            # Check for music-like patterns (simplified)
            # This would be more sophisticated with actual music detection
            return "music_or_noise"

    def _extract_audio_concepts(self, audio: AudioSegment, transcription: Dict,
                               context: Optional[Dict] = None) -> List[Dict]:
        """Extract semantic concepts from audio content."""
        concepts = []

        # Extract concepts from transcription text
        if transcription.get('success') and transcription.get('text'):
            text = transcription['text']

            # Simple keyword extraction (would be enhanced with NLP)
            keywords = self._extract_keywords_from_text(text)

            for keyword in keywords:
                concepts.append({
                    'concept': keyword,
                    'category': 'spoken_content',
                    'confidence': 0.7,
                    'source': 'transcription_analysis'
                })

        # Audio-based concepts (tone, pace, etc.)
        audio_features = self._extract_audio_features(audio)

        # Voice characteristics
        if audio_features['zero_crossing_rate'] > 0.1:  # High frequency content
            concepts.append({
                'concept': 'high_pitch_voice',
                'category': 'voice_characteristics',
                'confidence': 0.6,
                'source': 'audio_analysis'
            })

        # Speech patterns
        content_analysis = self._analyze_audio_content(audio)
        if content_analysis['speech_percentage'] > 80:
            concepts.append({
                'concept': 'continuous_speech',
                'category': 'speech_pattern',
                'confidence': 0.8,
                'source': 'audio_analysis'
            })

        return concepts[:10]  # Return top 10 concepts

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from transcribed text."""
        # Simple keyword extraction - would be enhanced with proper NLP
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        keywords = []
        for word in words:
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 3 and word not in stop_words:
                keywords.append(word)

        # Return unique keywords
        return list(set(keywords))[:5]

    def _analyze_emotion_and_sentiment(self, audio: AudioSegment, transcription: Dict) -> Dict[str, Any]:
        """Analyze emotion and sentiment from audio and transcription."""
        # This is a simplified implementation
        # Real emotion detection would use acoustic features and ML models

        emotion_result = {
            'detected_emotion': 'neutral',
            'confidence': 0.5,
            'sentiment': 'neutral',
            'intensity': 0.5
        }

        # Basic emotion detection from audio features
        features = self._extract_audio_features(audio)

        # Simple heuristics based on audio features
        if features['zero_crossing_rate'] > 0.15:  # High pitch variation
            emotion_result.update({
                'detected_emotion': 'excited',
                'confidence': 0.6,
                'intensity': 0.7
            })
        elif features['rms_amplitude'] < 5000:  # Quiet voice
            emotion_result.update({
                'detected_emotion': 'calm',
                'confidence': 0.6,
                'intensity': 0.4
            })

        # Sentiment from transcription (very basic)
        if transcription.get('success') and transcription.get('text'):
            text = transcription['text'].lower()
            positive_words = {'good', 'great', 'excellent', 'happy', 'love', 'wonderful'}
            negative_words = {'bad', 'terrible', 'hate', 'sad', 'angry', 'awful'}

            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)

            if positive_count > negative_count:
                emotion_result['sentiment'] = 'positive'
            elif negative_count > positive_count:
                emotion_result['sentiment'] = 'negative'

        return emotion_result

    def _cache_result(self, signature: str, result: Dict):
        """Cache processing result."""
        if len(self.audio_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.audio_cache))
            del self.audio_cache[oldest_key]

        self.audio_cache[signature] = result.copy()

    def compare_audio(self, audio1: Any, audio2: Any) -> Dict[str, Any]:
        """Compare two audio samples for similarity."""
        audio_seg1 = self._load_audio(audio1)
        audio_seg2 = self._load_audio(audio2)

        if audio_seg1 is None or audio_seg2 is None:
            return {"error": "Failed to load one or both audio samples"}

        try:
            # Simple similarity based on audio features
            features1 = self._extract_audio_features(audio_seg1)
            features2 = self._extract_audio_features(audio_seg2)

            # Compare key features
            rms_diff = abs(features1['rms_amplitude'] - features2['rms_amplitude'])
            zcr_diff = abs(features1['zero_crossing_rate'] - features2['zero_crossing_rate'])

            # Simple similarity score
            similarity = 1.0 - min(1.0, (rms_diff / 10000 + zcr_diff) / 2)

            return {
                'similarity_score': float(similarity),
                'similarity_percentage': float(similarity * 100),
                'are_similar': similarity > 0.7
            }

        except Exception as e:
            logger.error(f"Audio comparison failed: {e}")
            return {"error": f"Comparison failed: {str(e)}"}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the audio processing cache."""
        return {
            'cached_audio': len(self.audio_cache),
            'max_cache_size': self.cache_max_size,
            'cache_utilization': len(self.audio_cache) / self.cache_max_size if self.cache_max_size > 0 else 0.0
        }

    def record_audio(self, duration_seconds: int = 5, save_path: Optional[str] = None) -> Optional[AudioSegment]:
        """
        Record audio from microphone.

        Args:
            duration_seconds: Recording duration
            save_path: Optional path to save recording

        Returns:
            AudioSegment of recorded audio
        """
        try:
            with sr.Microphone() as source:
                logger.info(f"Recording audio for {duration_seconds} seconds...")
                self.recognizer.adjust_for_ambient_noise(source)

                audio_data = self.recognizer.record(source, duration=duration_seconds)

                # Convert to AudioSegment
                wav_data = io.BytesIO()
                wav_file = wave.open(wav_data, 'wb')
                wav_file.setnchannels(audio_data.channels)
                wav_file.setsampwidth(audio_data.sample_width)
                wav_file.setframerate(audio_data.sample_rate)
                wav_file.writeframes(audio_data.frame_data)
                wav_file.close()
                wav_data.seek(0)

                audio_segment = AudioSegment.from_wav(wav_data)

                if save_path:
                    audio_segment.export(save_path, format='wav')
                    logger.info(f"Audio saved to {save_path}")

                return audio_segment

        except Exception as e:
            logger.error(f"Audio recording failed: {e}")
            return None

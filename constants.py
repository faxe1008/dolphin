# Number of channels for live sampling
CHANNELS = 2
# Expected sample rate
RATE = 44100
# Size of a sampling chunk
CHUNK = RATE

# Number of Features after extraction
FEATURE_COUNT = 173
# Target class count
CLASS_COUNT = 2

# Librosa hop length
HOP_LENGTH = 512
# FFT Window size
NFFT = 2048
# Number of seconds that are sampled
SAMPLE_LENGTH_SEC = 2
# Number of melspectrogram buckets to calculate
N_MELS = 256

# Input dimension of CNN
INPUT_DIMENSION = (N_MELS, 173, 1)
# Output dimension of CNN
OUTPUT_DIMENSION = CLASS_COUNT

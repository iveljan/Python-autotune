import librosa

class IO:
    FILE_PATH = ''
    SAMPLE_RATE = 44100

    def __init__(self, file_path):
        self.FILE_PATH = file_path

    def load(self):
        if ".wav" in self.FILE_PATH:
            import wave
            with wave.open(self.FILE_PATH,'rb') as wav:
                self.SAMPLE_RATE = wav.getframerate()
        try:
            y, sr = librosa.load(self.FILE_PATH, sr=self.SAMPLE_RATE)
            return (y, sr)
        except:
            print("Error loading file: %s" % self.FILE_PATH)

    def dump(self, file_name, n_arr, sr):
        try:
            import soundfile as sf
            sf.write(file_name, n_arr, sr)
            print("Uspjeh dumping file: %s"%file_name)

        except:
            print("Error dumping file: %s"%file_name)


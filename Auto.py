import librosa
import Notes
import numpy as np


class Autotune:
    def __init__(self, y, sr, scale):
        self.INPUT_WAVE = y
        self.INPUT_SR = sr
        self.SCALE = scale
        self._note = Notes.Notes(scale)

        self.NOTES = self._note.getScale()
        self.OUTPUT_WAVE = np.empty(shape=self.INPUT_WAVE.shape)

    def correct(self):
        step = int(self.INPUT_SR / 20)
        print('Detected fq\tCorrected fq\tCorrection factor')
        print('--------------------------------------------')
        for x in range(0, len(self.INPUT_WAVE), step):
            y = self.INPUT_WAVE[x:x + step]
            f = self._findStft(y)

            diff_array = [np.abs(note - f) for note in self.NOTES]
            note = np.argmin(diff_array)
            print(f, end='\t')
            print(self.NOTES[note], end='\t')

            self.OUTPUT_WAVE[x:x + step] = self._transpose(y, f, self.NOTES[note])

        print('-------------------------------------------')
        return librosa.util.normalize(self.OUTPUT_WAVE)

    def _findStft(self, y):
        # Adjust n_fft to be appropriate for the input length
        n_fft = min(len(y), self.INPUT_SR)
        yD = librosa.stft(y, n_fft=n_fft)
        arr = np.argmax(yD, axis=0)
        fq = np.mean(arr)

        return self._note.normalize(fq)

    def _transpose(self, y, fold, fnew):
        # Calculate the steps to be transposed based on the old and new frequencies.
        steps = self._note.getStep(fold, fnew)
        print(steps)
        # Ensure the correct call to pitch_shift
        yT = librosa.effects.pitch_shift(y, sr=self.INPUT_SR, n_steps=steps)
        return yT

# Beat tracking example
from __future__ import print_function
import librosa
import matplotlib.pyplot as plt

# 1. Get the file path to the included audio example
cartoon_list = librosa.util.find_files("/home/raph/PycharmProjects/psc/src")
cartoon = cartoon_list[0]
# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(cartoon)
print (y)
# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print('Saving output to beat_times.csv')
librosa.output.times_csv('beat_times.csv', beat_times)

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)


librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()



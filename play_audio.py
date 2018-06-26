import pyaudio
import wave
import sys

def playback(input_file):
    """
    This function plays back the audio file on the default output
    It is self contained 
    """
    # metadata 
    CHUNK = 1000 

    # wavefile read obj + PyAudio obj
    wf = wave.open(input_file, "rb")
    p_playback = pyaudio.PyAudio() 

    # playback file obj ; note output=True
    stream_playback = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

    TIME_start = time.time()
    data = wf.readframes(CHUNK)
    while len(data) != 0:
        stream_playback.write(data)
        data = wf.readframes(CHUNK)
    TIME_stop = time.time()
    
    # Clean Up
    stream_playback.stop_stream()
    stream_playback.close() 
    p.terminate()

    return TIME_start, TIME_stop


def record(RECORD_SECONDS):
    """
    Function to record audio, returns a list of frames which must be manually dumped to a wav file 
    """

    # meta 
    RATE = 16000 
    CHUNK = 1000 
    CHANNELS = 1 
    FORMAT = pyaudio.paInt16

    p_record = pyaudio.PyAudio() 

    stream_record = p_record.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream_record.read(CHUNK)
            frames.append(data)

    print("* done recording")

    stream_record.stop_stream()
    stream_record.close()
    p_record.terminate()

    return frames, b''.join(frames)

def test(): 

    # test record 
    frames, samples = record(RECORD_SECONDS=5)

    wf = wave.open("recorded.wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(samples)
    wf.close()


if __name__ == "__main__":
    test() 

import pyaudio
import wave
import sys
from multiprocessing import Process

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

    TIME_start = time.time()
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
    TIME_stop = time.time()

    return b''.join(frames), TIME_start, TIME_stop

def play_record_multi(input_file, output_file):

    PROCESS_record = Process(target=record, args=(3,))
    PROCESS_record.start() 

    # playback 
    PLAYBACK_start, PLAYBACK_stop = playback(input_file)

    # wait for recording to finish 
    joined_frames, RECORD_start, RECORD_stop = PROCESS_record.join()

    # Clip joined_frames from PLAYBACK_start to PLAYBACK_stop
    # TODO 

    # save the final 
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(samples)
    wf.close()
    

if __name__ == "__main__":

    print(
    """Edit this file to call play_record_multi with 
    argument 1: wav file to be played 
    argument 2: recorded wav to be saved to disk 
    e.g play_record_multi("yes.wav", "recorded_yes.wav")
    """
    )

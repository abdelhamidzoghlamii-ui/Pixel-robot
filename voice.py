import subprocess, os, time
HOME = os.path.expanduser("~")
WHISPER_BIN = HOME + "/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = HOME + "/whisper.cpp/models/ggml-base.bin"
RAW_FILE = HOME + "/rec_raw.amr"
WAV_FILE = HOME + "/rec.wav"

def listen(duration=5):
    for f in [RAW_FILE, WAV_FILE]:
        if os.path.exists(f): os.remove(f)
    subprocess.Popen(["termux-microphone-record", "-f", RAW_FILE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[MIC] Recording " + str(duration) + "s... speak now!")
    time.sleep(duration)
    subprocess.run(["termux-microphone-record", "-q"], capture_output=True)
    time.sleep(0.5)
    if not os.path.exists(RAW_FILE) or os.path.getsize(RAW_FILE) < 1000: return ""
    subprocess.run(["ffmpeg","-i",RAW_FILE,"-ar","16000","-ac","1","-c:a","pcm_s16le",WAV_FILE,"-y"], capture_output=True)
    if not os.path.exists(WAV_FILE) or os.path.getsize(WAV_FILE) < 1000: return ""
    print("[MIC] Transcribing...")
    r = subprocess.run([WHISPER_BIN,"-m",WHISPER_MODEL,"-f",WAV_FILE,"--no-timestamps","-l","en"], capture_output=True, text=True)
    return " ".join(l.strip() for l in r.stdout.splitlines() if l.strip() and not l.strip().startswith("[") and not l.strip().startswith("whisper"))

if __name__ == "__main__":
    print("Say a robot command...")
    print("Heard: [" + listen(5) + "]")

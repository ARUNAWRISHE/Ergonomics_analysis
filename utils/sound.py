# utils/sound.py
import sys
import threading

try:
    import simpleaudio as sa
    HAVE_SIMPLEAUDIO = True
except ImportError:
    HAVE_SIMPLEAUDIO = False

def _beep_terminal():
    sys.stdout.write("\a")
    sys.stdout.flush()

def beep(sound_file: str | None = None):
    """
    Non-blocking beep.
    - If simpleaudio is available and a WAV path is provided, plays it.
    - Else falls back to terminal bell.
    """
    def _play():
        try:
            if HAVE_SIMPLEAUDIO and sound_file:
                wave_obj = sa.WaveObject.from_wave_file(sound_file)
                wave_obj.play()
            else:
                _beep_terminal()
        except Exception:
            _beep_terminal()
    threading.Thread(target=_play, daemon=True).start()

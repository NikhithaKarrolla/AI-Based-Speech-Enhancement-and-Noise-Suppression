import os
import soundfile as sf


def check_audio_file(file_path: str):

    print("\nChecking audio file...")
    print("File:", file_path)

    if not os.path.exists(file_path):
        print("ERROR: File does not exist.")
        return False

    print(
        "File size:",
        os.path.getsize(file_path),
        "bytes"
    )

    try:

        info = sf.info(file_path)

        print("\nAudio information:")
        print("Format:", info.format)
        print("Subtype:", info.subtype)
        print("Sample rate:", info.samplerate)
        print("Channels:", info.channels)
        print("Duration:", info.duration)

        return True

    except Exception as e:

        print("\nERROR: Audio format could not be read.")
        print("Reason:", e)

        return False
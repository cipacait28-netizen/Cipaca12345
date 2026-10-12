# toon-video

Beat-synced 1930s-cartoon / halftone dot-art photo video, drawn in code (numpy + OpenCV + ffmpeg).

Put photos in `assets/photos/p1..p9` and the song at `assets/song.mp3` (both git-ignored), then:

    pip install numpy pillow opencv-python-headless scipy
    python analyze_audio.py
    python make_video.py stills 3 10 40     # preview frames
    python make_video.py render             # -> out/video.mp4

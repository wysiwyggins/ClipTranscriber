import os
import subprocess
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import whisper

# ---- Configuration ----
video_path = "assets/videoplayback.mp4"
model_size = "medium"  # Whisper model size
scale_height = 320
crop_width = 460
crop_height = 320

# ---- Step 1: Detect Scenes ----
video_manager = VideoManager([video_path])
scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=30.0))
video_manager.start()
scene_manager.detect_scenes(frame_source=video_manager)
scene_list = scene_manager.get_scene_list()

# ---- Step 2: Split, Resize, and Crop ----
basename, ext = os.path.splitext(video_path)
for i, scene in enumerate(scene_list):
    start_seconds = scene[0].get_seconds()
    end_seconds = scene[1].get_seconds()
    duration = end_seconds - start_seconds

    out_file = f"{basename}_part_{i}{ext}"

    # Applying filters:
    # - Scale height to 320px, width automatically
    # - Then crop width to 460px, height to 320px
    vf_filter = f"scale=-1:{scale_height},crop={crop_width}:{crop_height}"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_seconds),
        "-t", str(duration),
        "-i", video_path,
        "-vf", vf_filter,
        "-c:a", "copy",  # copy audio as-is
        out_file
    ]
    subprocess.run(cmd, check=True)

# ---- Step 3: Extract Audio and Transcribe ----
model = whisper.load_model(model_size)

for i, scene in enumerate(scene_list):
    clip_file = f"{basename}_part_{i}{ext}"
    audio_file = f"{basename}_part_{i}.wav"
    # Extract audio
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", clip_file,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_file
    ]
    subprocess.run(cmd, check=True)
    
    # Transcribe using Whisper
    result = model.transcribe(audio_file)
    segments = result["segments"]
    
    # ---- Step 4: Write VTT ----
    vtt_file = f"{basename}_part_{i}.vtt"
    with open(vtt_file, "w") as vtt:
        vtt.write("WEBVTT\n\n")
        
        for seg in segments:
            start_time = seg["start"]
            end_time = seg["end"]
            text = seg["text"].strip()
            
            def to_vtt_time(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = seconds % 60
                return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')
            
            vtt.write(f"{to_vtt_time(start_time)} --> {to_vtt_time(end_time)}\n{text}\n\n")

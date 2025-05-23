import os
import tempfile
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import speech_recognition as sr
from pydub import AudioSegment
from pydub.generators import Sine
from transformers import pipeline
from typing import Optional, Tuple
import uuid

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'audio_uploads'  # Use a persistent directory
ALLOWED_EXTENSIONS = {'wav'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Initialize components
try:
    toxicity_classifier = pipeline(
        "text-classification", 
        model="fatmhd1995/roberta-hate-speech-dynabench-r4-target-TOXICITY-FT"
    )
    beep = Sine(1000).to_audio_segment(duration=100)
except Exception as e:
    print(f"Error initializing components: {e}")
    toxicity_classifier = None
    beep = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_audio_file(audio: AudioSegment, prefix: str = '') -> str:
    """Create an audio file and return its path."""
    filename = f"{prefix}{str(uuid.uuid4())}.wav"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio.export(filepath, format="wav")
    return filename  # Return just the filename, not full path

def load_and_transcribe_audio(wav_path: str) -> Tuple[Optional[str], Optional[AudioSegment]]:
    """Load WAV file and transcribe using Google/Sphinx with word timings."""
    recognizer = sr.Recognizer()
    try:
        audio = AudioSegment.from_wav(wav_path)
        
        # Create a temporary file for speech recognition
        temp_filename = create_audio_file(audio, "temp_")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        try:
            with sr.AudioFile(temp_path) as source:
                audio_data = recognizer.record(source)
                result = recognizer.recognize_google(audio_data, show_all=True)
                
                if isinstance(result, dict) and 'alternative' in result:
                    if 'words' in result['alternative'][0]:
                        return result['alternative'][0]['words'], audio
                    return result['alternative'][0]['transcript'], audio
                return result, audio
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    except Exception as e:
        print(f"Transcription error: {e}")
        return None, None

def is_toxic(word: str) -> bool:
    """Check if word is toxic."""
    if not word.strip() or not toxicity_classifier:
        return False
    try:
        result = toxicity_classifier(word)[0]
        return result["label"] == "Toxic" and result["score"] > 0.7
    except:
        return False

def censor_audio(words_info, audio: AudioSegment) -> Optional[AudioSegment]:
    """Censor toxic words in audio."""
    if not beep:
        return None
        
    censored = AudioSegment.silent(duration=0)
    last_end = 0
    
    if isinstance(words_info, list):  # Has word timings
        for word in words_info:
            start = int(word['start_time'] * 1000)
            end = int(word['end_time'] * 1000)
            
            if last_end < start:
                censored += audio[last_end:start]
            
            censored += beep if is_toxic(word['word']) else audio[start:end]
            last_end = end
        
        if last_end < len(audio):
            censored += audio[last_end:]
    elif isinstance(words_info, str):  # Only text
        words = words_info.split()
        duration = len(audio) / len(words)
        
        for i, word in enumerate(words):
            start = int(i * duration)
            end = int((i + 1) * duration)
            censored += beep if is_toxic(word) else audio[start:end]
    
    return censored

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Create a unique filename for the upload
            original_filename = f"original_{str(uuid.uuid4())}.wav"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            file.save(upload_path)
            
            try:
                words_info, audio = load_and_transcribe_audio(upload_path)
                if not words_info or not audio:
                    flash('Error processing audio')
                    try:
                        os.unlink(upload_path)
                    except:
                        pass
                    return redirect(request.url)
                
                censored = censor_audio(words_info, audio)
                if not censored:
                    flash('Error generating censored audio')
                    try:
                        os.unlink(upload_path)
                    except:
                        pass
                    return redirect(request.url)
                
                # Create processed file
                processed_filename = create_audio_file(censored, "processed_")
                
                return render_template('index.html', 
                                    original=original_filename,
                                    processed=processed_filename)
            
            except Exception as e:
                flash(f'Error: {str(e)}')
                try:
                    os.unlink(upload_path)
                except:
                    pass
                return redirect(request.url)
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        response = send_file(
            file_path,
            as_attachment=True,
            download_name=filename
        )
        
        # Schedule file for cleanup after download
        @response.call_on_close
        def cleanup():
            try:
                os.unlink(file_path)
                # Also try to delete the counterpart file
                counterpart = ("original_" if filename.startswith("processed_") else "processed_") + filename.split("_", 1)[1]
                counterpart_path = os.path.join(app.config['UPLOAD_FOLDER'], counterpart)
                if os.path.exists(counterpart_path):
                    os.unlink(counterpart_path)
            except Exception as e:
                print(f"Error cleaning up files: {e}")
        
        return response
    except FileNotFoundError:
        flash('File not found or already downloaded')
        return redirect(url_for('upload_file'))

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """Endpoint to manually clean up old files"""
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        return "Cleanup complete", 200
    except Exception as e:
        return f"Cleanup failed: {str(e)}", 500

if __name__ == '__main__':
    try:
        app.run()
    finally:
        # Optional: Add cleanup on server stop if desired
        pass

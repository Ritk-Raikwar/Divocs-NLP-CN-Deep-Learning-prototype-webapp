from flask import Flask, render_template, request, jsonify, url_for, redirect, flash
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
import tempfile
import uuid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import transformers
import shutil
# Import custom model components
from models.emotion_recognition import EmotionRecognitionModel
from models.clip_model import CLIPModel
#from models.slide_extractor import SlideExtractor

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'pdf', 'txt', 'jpg', 'jpeg', 'png'}

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)
for subfolder in ['text', 'dubbed', 'summaries', 'videos', 'ocr']:
    os.makedirs(os.path.join(app.config['RESULTS_FOLDER'], subfolder), exist_ok=True)

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize face detection model
try:
    face_detector = MTCNN(keep_all=True, device=device)
except Exception as e:
    print(f"Error loading MTCNN face detector: {str(e)}")
    face_detector = None

# Initialize emotion recognition model
emotion_model_path = "models/emotion_recognition/emotion_recognition.pth"
try:
    if os.path.exists(emotion_model_path):
        emotion_model = EmotionRecognitionModel(model_path=emotion_model_path, device=device)
    else:
        print(f"Warning: Emotion model file '{emotion_model_path}' not found. Using untrained model.")
        emotion_model = EmotionRecognitionModel(model_path=None, device=device)
except Exception as e:
    print(f"Error loading EmotionRecognitionModel: {str(e)}")
    emotion_model = None

# Initialize CLIP model
try:
    clip_model = CLIPModel(device=device)
    if clip_model.model is None:
        print("Warning: CLIP model failed to load. Scene analysis will be limited.")
except Exception as e:
    print(f"Error loading CLIPModel: {str(e)}")
    clip_model = None

# Emotion labels (global constant)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_frames(video_path, num_frames=None, fps=1):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / original_fps if original_fps > 0 else 0.0

    print(f"Video info: {total_frames} frames, {original_fps} fps, {duration:.2f} seconds")

    if num_frames is None:
        frames_to_extract = int(duration * fps)
        frame_indices = np.linspace(0, total_frames - 1, frames_to_extract, dtype=int)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    timestamps = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        timestamp = frame_idx / original_fps
        timestamps.append(timestamp)

    cap.release()
    return frames, timestamps

def process_frame(frame, frame_idx, analysis_id, face_detector, emotion_model, clip_model):
    results = {}

    frame_path = os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}_frame_{frame_idx}.png")
    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    results['image_url'] = f"results/{analysis_id}_frame_{frame_idx}.png"

    pil_img = Image.fromarray(frame)

    if face_detector is None:
        results['face_count'] = 0
        results['emotions'] = {}
        results['primary_emotion'] = 'Face detection unavailable'
        results['emotion_probs'] = {emotion: 0.0 for emotion in EMOTION_LABELS}
    else:
        boxes, _ = face_detector.detect(pil_img)
        if boxes is not None and len(boxes) > 0:
            faces = []
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face_resized = cv2.resize(face, (48, 48))
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
                faces.append(face_gray)

            results['face_count'] = len(faces)

            if faces and emotion_model:
                emotions = []
                emotion_probs = []
                for face in faces:
                    probs = emotion_model.predict(face)
                    emotion_idx = torch.argmax(probs).item()
                    emotion_name = EMOTION_LABELS[emotion_idx]
                    emotions.append(emotion_name)
                    emotion_probs.append(probs.tolist())

                if emotion_probs:
                    avg_probs = np.mean(emotion_probs, axis=0)
                    results['emotion_probs'] = {EMOTION_LABELS[i]: prob for i, prob in enumerate(avg_probs)}

                emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}
                results['emotions'] = emotion_counts

                if emotions:
                    most_common = max(set(emotions), key=emotions.count)
                    results['primary_emotion'] = most_common
                else:
                    results['primary_emotion'] = 'Unknown'
            else:
                results['emotions'] = {}
                results['primary_emotion'] = 'Unknown'
                results['emotion_probs'] = {emotion: 0.0 for emotion in EMOTION_LABELS}
        else:
            results['face_count'] = 0
            results['emotions'] = {}
            results['primary_emotion'] = 'No faces detected'
            results['emotion_probs'] = {emotion: 0.0 for emotion in EMOTION_LABELS}

    scene_labels = ['indoor', 'outdoor', 'dark', 'bright', 'crowded', 'empty', 'professional', 'casual', 'formal', 'social gathering']
    if clip_model and clip_model.model:
        scene_scores = clip_model.analyze_image(frame, scene_labels)
        top_scenes = [(scene_labels[i], float(score)) for i, score in enumerate(scene_scores.tolist()) if score > 0.2]
        results['scene_context'] = dict(top_scenes)
    else:
        results['scene_context'] = {}

    return results

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum size allowed is 300MB.')
    return redirect(url_for('video_sentiment')), 413

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/services')
def services():
    return render_template('service.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_sentiment', methods=['GET'])
def video_sentiment():
    if not face_detector or not emotion_model:
        flash('Sentiment analysis is currently unavailable due to model loading issues.')
        return redirect(url_for('index'))
    return render_template('upload.html', feature='Video Sentiment Analysis')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('No video file uploaded')
        return redirect(url_for('video_sentiment'))

    video_file = request.files['video']
    if video_file.filename == '':
        flash('No video selected')
        return redirect(url_for('video_sentiment'))

    if not allowed_file(video_file.filename):
        flash('Invalid video format. Allowed formats: mp4, avi, mov, mkv')
        return redirect(url_for('video_sentiment'))

    analysis_id = str(uuid.uuid4())
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(temp_dir, filename)
        video_file.save(video_path)
        video_filename = filename

        print(f"Video saved at: {video_path}")
        if not os.path.exists(video_path):
            flash('Failed to save video file')
            return redirect(url_for('video_sentiment'))

        start_time = time.time()
        frames, timestamps = extract_frames(video_path, fps=1)

        if not frames:
            print("No frames extracted")
            flash('Could not extract frames from video')
            return redirect(url_for('video_sentiment'))

        print(f"Extracted {len(frames)} frames")
        frame_results = []
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            print(f"Processing frame {i+1}/{len(frames)}")
            result = process_frame(frame, i, analysis_id, face_detector, emotion_model, clip_model)
            result['timestamp'] = timestamp
            frame_results.append(result)

        if not frame_results:
            print("No frame results generated")
            flash('No frame results generated')
            return redirect(url_for('video_sentiment'))

        total_frames = len(frame_results)
        face_frames = sum(1 for r in frame_results if r['face_count'] > 0)

        all_emotions = []
        emotion_prob_sums = {emotion: 0.0 for emotion in EMOTION_LABELS}
        for r in frame_results:
            emotions_in_frame = r.get('emotions', {})
            for emotion, count in emotions_in_frame.items():
                all_emotions.extend([emotion] * count)
            probs = r.get('emotion_probs', {})
            for emotion, prob in probs.items():
                emotion_prob_sums[emotion] += prob

        emotion_summary = {}
        total_emotion_frames = sum(1 for r in frame_results if r.get('emotions', {}))
        if all_emotions and total_emotion_frames > 0:
            for emotion in EMOTION_LABELS:
                prob_avg = emotion_prob_sums[emotion] / total_emotion_frames
                emotion_summary[emotion] = prob_avg * 100

        scene_contexts = {}
        for r in frame_results:
            for scene, score in r.get('scene_context', {}).items():
                if scene not in scene_contexts:
                    scene_contexts[scene] = []
                scene_contexts[scene].append(score)

        scene_summary = {scene: sum(scores)/len(scores) * 100 for scene, scores in scene_contexts.items() if scores}

        timeline_data = []
        for r in frame_results:
            primary_emotion = r.get('primary_emotion', 'Unknown')
            if primary_emotion not in ['No faces detected', 'Unknown', 'Face detection unavailable']:
                intensity = r.get('emotion_probs', {}).get(primary_emotion, 0.0)
                timeline_data.append({
                    'timestamp': r['timestamp'],
                    'emotion': primary_emotion,
                    'intensity': intensity
                })

        summary = {
            'analysis_id': analysis_id,
            'video_filename': filename,
            'duration': timestamps[-1] if timestamps else 0,
            'frames_analyzed': total_frames,
            'frames_with_faces': face_frames,
            'face_detection_rate': (face_frames / total_frames * 100) if total_frames > 0 else 0,
            'emotion_distribution': emotion_summary,
            'scene_context': scene_summary,
            'emotional_timeline': timeline_data,
            'analysis_duration': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }

        results_path = os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'summary': summary,
                'frame_details': frame_results
            }, f, indent=2)

        create_visualizations(summary, analysis_id)

        chart_emotion_labels = list(emotion_summary.keys()) if emotion_summary else EMOTION_LABELS
        chart_emotion_values = [v for v in emotion_summary.values()] if emotion_summary else [0] * len(EMOTION_LABELS)
        # Filter out zero values for better visualization
        if emotion_summary:
            filtered_emotions = {k: v for k, v in emotion_summary.items() if v > 0}
            chart_emotion_labels = list(filtered_emotions.keys()) or ['No Data']
            chart_emotion_values = list(filtered_emotions.values()) or [100]

        chart_content_labels = list(scene_summary.keys()) if scene_summary else ['No Data']
        chart_content_values = list(scene_summary.values()) if scene_summary else [0]
        if scene_summary:
            filtered_content = {k: v for k, v in scene_summary.items() if v > 0}
            chart_content_labels = list(filtered_content.keys()) or ['No Data']
            chart_content_values = list(filtered_content.values()) or [0]

        chart_timeline_labels = [item['timestamp'] for item in timeline_data] if timeline_data else [0]
        chart_emotion_timeline = timeline_data if timeline_data else [{'timestamp': 0, 'emotion': 'Unknown', 'intensity': 0}]

        formatted_frame_details = []
        for i, frame in enumerate(frame_results):
            emotions = frame.get('emotions', {})
            total_faces = sum(emotions.values(), 0)
            normalized_emotions = {emotion: count / total_faces if total_faces > 0 else 0.0 for emotion, count in emotions.items()}
            formatted_frame = {
                'time': f"{frame['timestamp']:.2f}s",
                'emotions': normalized_emotions,
                'content': frame.get('scene_context', {}),
                'image_url': url_for('static', filename=frame['image_url']),
                'primary_emotion': frame.get('primary_emotion', 'Unknown'),
                'face_count': frame.get('face_count', 0),
                'timestamp': frame['timestamp']
            }
            formatted_frame_details.append(formatted_frame)

        return render_template('results.html',
                              summary=summary,
                              vis_files={
                                  'emotion_pie': os.path.exists(os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}_emotion_pie.png")),
                                  'emotion_timeline': os.path.exists(os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}_emotion_timeline.png")),
                                  'scene_context': os.path.exists(os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}_scene_context.png"))
                              },
                              analysis_id=analysis_id,
                              video_path=video_filename,
                              video_filename=video_filename,
                              emotion_labels=chart_emotion_labels,
                              emotion_values=chart_emotion_values,
                              content_labels=chart_content_labels,
                              content_values=chart_content_values,
                              timeline_labels=chart_timeline_labels,
                              emotion_timeline=chart_emotion_timeline,
                              topics=[(label, value/100) for label, value in zip(chart_content_labels, chart_content_values)],
                              emotions=emotion_summary,
                              primary_emotion=max(emotion_summary.items(), key=lambda x: x[1])[0] if emotion_summary else "Unknown",
                              dominant_theme=max(scene_summary.items(), key=lambda x: x[1])[0] if scene_summary else "Unknown",
                              face_count=int(round(face_frames / total_frames * summary.get('frames_analyzed', 1))) if total_frames > 0 else 0,
                              frame_details=formatted_frame_details)

    except Exception as e:
        import traceback
        print(f"Error processing video: {str(e)}")
        traceback.print_exc()
        flash(f'Error processing video: {str(e)}')
        return redirect(url_for('video_sentiment'))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def create_visualizations(summary, analysis_id):
    if summary['emotion_distribution']:
        plt.figure(figsize=(10, 6))
        emotions = list(summary['emotion_distribution'].keys())
        values = list(summary['emotion_distribution'].values())
        plt.pie(values, labels=emotions, autopct='%1.1f%%')
        plt.title('Emotion Distribution')
        plt.savefig(os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}_emotion_pie.png"))
        plt.close()

    if summary['emotional_timeline']:
        plt.figure(figsize=(12, 6))
        timestamps = [item['timestamp'] for item in summary['emotional_timeline']]
        emotions = [item['emotion'] for item in summary['emotional_timeline']]
        unique_emotions = list(set(emotions))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_emotions)))
        emotion_colors = {emotion: colors[i] for i, emotion in enumerate(unique_emotions)}

        for emotion in unique_emotions:
            emotion_indices = [i for i, e in enumerate(emotions) if e == emotion]
            emotion_timestamps = [timestamps[i] for i in emotion_indices]
            emotion_y = [unique_emotions.index(emotion)] * len(emotion_indices)
            plt.scatter(emotion_timestamps, emotion_y, label=emotion, color=emotion_colors[emotion], s=50)

        if len(timestamps) > 1:
            plt.plot(timestamps, [unique_emotions.index(e) for e in emotions], color='gray', alpha=0.5, linestyle='-')

        plt.yticks(range(len(unique_emotions)), unique_emotions)
        plt.xlabel('Time (seconds)')
        plt.title('Emotional Timeline')
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}_emotion_timeline.png"))
        plt.close()

    if summary['scene_context']:
        plt.figure(figsize=(10, 6))
        scenes = list(summary['scene_context'].keys())
        values = list(summary['scene_context'].values())
        plt.barh(scenes, values)
        plt.xlabel('Confidence (%)')
        plt.title('Scene Context Analysis')
        plt.tight_layout()
        plt.savefig(os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}_scene_context.png"))
        plt.close()

@app.route('/results/<analysis_id>')
def results(analysis_id):
    results_path = os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}_results.json")
    if not os.path.exists(results_path):
        flash('Analysis results not found')
        return redirect(url_for('index'))

    with open(results_path, 'r') as f:
        data = json.load(f)

    summary = data['summary']
    frame_details = data['frame_details']

    vis_files = {
        'emotion_pie': os.path.exists(os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}_emotion_pie.png")),
        'emotion_timeline': os.path.exists(os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}_emotion_timeline.png")),
        'scene_context': os.path.exists(os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}_scene_context.png"))
    }

    video_filename = summary['video_filename']
    video_path = video_filename

    chart_emotion_labels = list(summary.get('emotion_distribution', {}).keys())
    chart_emotion_values = list(summary.get('emotion_distribution', {}).values())
    if chart_emotion_values:
        filtered_emotions = {k: v for k, v in summary.get('emotion_distribution', {}).items() if v > 0}
        chart_emotion_labels = list(filtered_emotions.keys()) or ['No Data']
        chart_emotion_values = list(filtered_emotions.values()) or [100]

    chart_content_labels = list(summary.get('scene_context', {}).keys())
    chart_content_values = list(summary.get('scene_context', {}).values())
    if chart_content_values:
        filtered_content = {k: v for k, v in summary.get('scene_context', {}).items() if v > 0}
        chart_content_labels = list(filtered_content.keys()) or ['No Data']
        chart_content_values = list(filtered_content.values()) or [0]

    timeline_data = summary.get('emotional_timeline', [])
    chart_timeline_labels = [item.get('timestamp', 0) for item in timeline_data]
    chart_emotion_timeline = timeline_data if timeline_data else [{'timestamp': 0, 'emotion': 'Unknown', 'intensity': 0}]

    topics = [(label, value/100) for label, value in zip(chart_content_labels, chart_content_values)]
    emotions = summary.get('emotion_distribution', {})
    primary_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "Unknown"
    scene_context = summary.get('scene_context', {})
    dominant_theme = max(scene_context.items(), key=lambda x: x[1])[0] if scene_context else "Unknown"
    face_detection_rate = summary.get('face_detection_rate', 0)
    face_count = int(round(face_detection_rate / 100 * summary.get('frames_analyzed', 0)))

    formatted_frame_details = []
    for i, frame in enumerate(frame_details):
        emotions = frame.get('emotions', {})
        total_faces = sum(emotions.values(), 0)
        normalized_emotions = {emotion: count / total_faces if total_faces > 0 else 0.0 for emotion, count in emotions.items()}
        formatted_frame = {
            'time': f"{frame['timestamp']:.2f}s",
            'emotions': normalized_emotions,
            'content': frame.get('scene_context', {}),
            'image_url': url_for('static', filename=frame['image_url']),
            'primary_emotion': frame.get('primary_emotion', 'Unknown'),
            'face_count': frame.get('face_count', 0),
            'timestamp': frame['timestamp']
        }
        formatted_frame_details.append(formatted_frame)

    return render_template('results.html',
                          summary=summary,
                          vis_files=vis_files,
                          analysis_id=analysis_id,
                          video_path=video_path,
                          video_filename=video_path,
                          emotion_labels=chart_emotion_labels,
                          emotion_values=chart_emotion_values,
                          content_labels=chart_content_labels,
                          content_values=chart_content_values,
                          timeline_labels=chart_timeline_labels,
                          emotion_timeline=chart_emotion_timeline,
                          topics=topics,
                          emotions=emotions,
                          primary_emotion=primary_emotion,
                          dominant_theme=dominant_theme,
                          face_count=face_count,
                          frame_details=formatted_frame_details)

@app.route('/text_extraction', methods=['GET', 'POST'])
def text_extraction():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(url_for('text_extraction'))
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('text_extraction'))
        if file and allowed_file(file.filename):
            analysis_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)

            try:
                summary = {
                    'total_words': 150,
                    'key_terms': ['example', 'text', 'video', 'AI'],
                    'extracted_text': 'This is a sample extracted text from the video.',
                    'duration': 120.0,
                    'analysis_duration': 2.5,
                    'status': 'Under development - Simulated data'
                }
                text_path = os.path.join(app.config['RESULTS_FOLDER'], 'text', f"{analysis_id}_text.txt")
                with open(text_path, 'w') as f:
                    f.write(summary['extracted_text'])
                return render_template('results1.html',
                                      video_path=filename,
                                      video_filename=filename,
                                      summary=summary,
                                      analysis_id=analysis_id)
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        flash('Invalid file type. Allowed: mp4, avi, mov, mkv')
        return redirect(url_for('text_extraction'))
    return render_template('upload.html', feature='Text Extraction')

@app.route('/dubbing', methods=['GET', 'POST'])
def dubbing():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(url_for('dubbing'))
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('dubbing'))
        if file and allowed_file(file.filename):
            analysis_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)

            try:
                summary = {
                    'target_language': 'Spanish',
                    'voice_match': 95.5,
                    'duration': 120.0,
                    'analysis_duration': 3.0,
                    'waveform_data': [0.1, 0.3, 0.5, 0.2, 0.4],
                    'status': 'Under development - Simulated data'
                }
                dubbed_path = os.path.join(app.config['RESULTS_FOLDER'], 'dubbed', f"{analysis_id}_dubbed.mp4")
                with open(dubbed_path, 'w') as f:
                    f.write('Placeholder dubbed video')
                return render_template('results2.html',
                                      video_path=filename,
                                      video_filename=filename,
                                      summary=summary,
                                      analysis_id=analysis_id)
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        flash('Invalid file type. Allowed: mp4, avi, mov, mkv')
        return redirect(url_for('dubbing'))
    return render_template('upload.html', feature='Video Dubbing')

@app.route('/summarization', methods=['GET', 'POST'])
def summarization():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(url_for('summarization'))
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('summarization'))
        if file and allowed_file(file.filename):
            analysis_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)

            try:
                summary = {
                    'summary_text': 'This is a concise summary of the video content.',
                    'key_points': [{'time': 30, 'point': 'Introduction'}, {'time': 60, 'point': 'Main topic'}],
                    'duration': 120.0,
                    'analysis_duration': 2.8,
                    'status': 'Under development - Simulated data'
                }
                summary_path = os.path.join(app.config['RESULTS_FOLDER'], 'summaries', f"{analysis_id}_summary.txt")
                with open(summary_path, 'w') as f:
                    f.write(summary['summary_text'])
                return render_template('results3.html',
                                      video_path=filename,
                                      video_filename=filename,
                                      summary=summary,
                                      analysis_id=analysis_id)
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        flash('Invalid file type. Allowed: mp4, avi, mov, mkv')
        return redirect(url_for('summarization'))
    return render_template('upload.html', feature='Video Summarization')

@app.route('/document_to_video', methods=['GET', 'POST'])
def document_to_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(url_for('document_to_video'))
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('document_to_video'))
        if file and allowed_file(file.filename):
            analysis_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)

            try:
                summary = {
                    'ai_teacher': 'Virtual Instructor',
                    'sections': ['Introduction', 'Main Content', 'Conclusion'],
                    'duration': 180.0,
                    'analysis_duration': 4.0,
                    'status': 'Under development - Simulated data'
                }
                video_path = os.path.join(app.config['RESULTS_FOLDER'], 'videos', f"{analysis_id}_video.mp4")
                with open(video_path, 'w') as f:
                    f.write('Placeholder generated video')
                return render_template('results4.html',
                                      video_path=filename,
                                      video_filename=filename,
                                      summary=summary,
                                      analysis_id=analysis_id)
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        flash('Invalid file type. Allowed: pdf, txt')
        return redirect(url_for('document_to_video'))
    return render_template('upload.html', feature='Document to Video')

@app.route('/handwriting_recognition', methods=['GET', 'POST'])
def handwriting_recognition():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(url_for('handwriting_recognition'))
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('handwriting_recognition'))
        if file and allowed_file(file.filename):
            analysis_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)

            try:
                start_time = time.time()
                file_extension = filename.rsplit('.', 1)[1].lower()
                duration = 0.0

                output_dir = os.path.join(app.config['RESULTS_FOLDER'], 'ocr', analysis_id)
                os.makedirs(output_dir, exist_ok=True)
                extractor = SlideExtractor(video_path=file_path, output_dir=output_dir)

                if file_extension in {'mp4', 'avi', 'mov', 'mkv'}:
                    success = extractor.extract_slides()
                    if not success:
                        flash('Failed to extract slides from video')
                        return redirect(url_for('handwriting_recognition'))

                    cap = cv2.VideoCapture(file_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps if fps > 0 else 0.0
                    cap.release()

                elif file_extension in {'jpg', 'jpeg', 'png'}:
                    frame = cv2.imread(file_path)
                    if frame is None:
                        flash('Failed to load image')
                        return redirect(url_for('handwriting_recognition'))

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    preprocessed = extractor._preprocess_image(frame_rgb)
                    text = extractor._extract_text(preprocessed)

                    slide_filename = f"slide_0001.png"
                    slide_path = os.path.join(output_dir, slide_filename)
                    cv2.imwrite(slide_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    extractor.slide_texts.append((slide_filename, text))
                    duration = 0.0

                else:
                    flash('Unsupported file type for handwriting recognition')
                    return redirect(url_for('handwriting_recognition'))

                # Get the extracted results
                slide_results = extractor.get_results()
                if not slide_results:
                    flash('No slides extracted from the input')
                    return redirect(url_for('handwriting_recognition'))

                # Generate LaTeX content for PDF
                latex_content = []
                for i, (slide_filename, text) in enumerate(slide_results, 1):
                    # Escape special LaTeX characters
                    text = text.replace('&', '\&').replace('%', '\%').replace('$', '\$').replace('#', '\#')
                    text = text.replace('_', '\_').replace('{', '\{').replace('}', '\}').replace('~', '\textasciitilde')
                    text = text.replace('^', '\textasciicircum').replace('\\', '\textbackslash')

                    latex_content.append(f"\\subsection{{Slide {i}: {slide_filename}}}")
                    latex_content.append(f"\\paragraph{{Extracted Text:}}")
                    latex_content.append(f"{text}\n")

                # Read the base LaTeX template (assuming it's available as notes.tex)
                latex_base_path = "notes.tex"  # Adjust path if needed
                if not os.path.exists(latex_base_path):
                    flash('LaTeX template not found')
                    return redirect(url_for('handwriting_recognition'))

                with open(latex_base_path, 'r') as f:
                    latex_template = f.read()

                # Insert the slide content before the end of the document
                latex_document = latex_template.replace('\\end{document}', '\n'.join(latex_content) + '\n\\end{document}')

                # Save the generated LaTeX file
                latex_output_path = os.path.join(app.config['RESULTS_FOLDER'], 'ocr', f"{analysis_id}_notes.tex")
                with open(latex_output_path, 'w') as f:
                    f.write(latex_document)

                # The system will render the LaTeX file to PDF using latexmk
                # Return the path to the LaTeX file (rendering handled externally)
                return jsonify({
                    'latex_file': latex_output_path,
                    'message': 'Handwriting recognition completed. PDF will be generated.'
                })

            except Exception as e:
                import traceback
                print(f"Error processing file for handwriting recognition: {str(e)}")
                traceback.print_exc()
                flash(f'Error processing file: {str(e)}')
                return redirect(url_for('handwriting_recognition'))
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        flash('Invalid file type. Allowed: jpg, jpeg, png, mp4, avi, mov, mkv')
        return redirect(url_for('handwriting_recognition'))
    return render_template('upload.html', feature='Handwriting Recognition')

if __name__ == '__main__':
    app.run(debug=True)
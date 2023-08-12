import os
import traceback
import openai
from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for, session, jsonify
from flask_session import Session
from werkzeug.utils import secure_filename
from assembly_ai_transcriber import AssemblyAITranscriber
import logging
from logging.handlers import RotatingFileHandler

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'aac'}
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'

Session(app)

# Configure logging
log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, "app.log")

handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=10, encoding='utf-8')  # 10 MB file size limit, keep last 10 files
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
)
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data['messages']
    model = data['model']
    api_key = data['api_key'] 

    if not api_key: 
        logger.warning('API key not provided for chat request.')  # Log a warning when API key isn't provided
        return jsonify({"error": "API key not provided"}), 400
    
    openai.api_key = api_key  # Set the OpenAI API key

    logger.info('Received a chat request.')  # Log the received request
    logger.info('Selected model: %s', model)  # Log the selected model

    try:
        # TODO: Remember to remove this log statement before committing to GitHub as it's for test purposes only
        logger.info('Sending the following messages to OpenAI: %s', messages)  

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            # Add other optional parameters if needed
        )
        logger.info('ChatGPT response generated successfully.')  # Log successful generation of response
        return jsonify(response)
    except openai.error.AuthenticationError as e:
        logger.error('AuthenticationError during chat request: %s', str(e))  # Log error with exception detail
        return jsonify({"error": "Invalid API key provided. Details: " + str(e)}), 401
    except openai.error.RateLimitError as e:
        logger.error('RateLimitError during chat request: %s', str(e))  # Log error with exception detail
        return jsonify({"error": "Rate limit reached for requests. Details: " + str(e), "retry": True}), 429
    except openai.error.APIConnectionError as e:
        logger.error('APIConnectionError during chat request')  # Log error WITHOUT exception detail
        return jsonify({"error": "Connection error to ChatGPT servers. Details: " + str(e), "retry": True}), 500
    except openai.error.InvalidRequestError as e:
        logger.error('InvalidRequestError during chat request')  # Log error WITHOUT exception detail
        return jsonify({"error": "Your request was malformed or missing some required parameters. Details: " + str(e)}), 400
    except openai.error.APIError as e:
        logger.error('APIError during chat request')  # Log error WITHOUT exception detail
        return jsonify({"error": "The OpenAI servers had an error while processing your request. Details: " + str(e), "retry": True}), 500
    except Exception as e:
        logger.error('Unexpected error during chat request: %s', str(e))  # Log error with exception detail
        return jsonify({"error": "An unexpected error occurred. Details: " + str(e)}), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Check if the file has a valid extension
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(audio_file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    audio_file = request.files['file']
    api_key = request.form.get('api_key')  # Get the API key from the form data
    service = request.args.get('service', 'assemblyai')  # default to assemblyai if service is not provided

    if not api_key:
        return jsonify({"error": "API key not provided"}), 400

    transcriber = None
    if service == 'assemblyai':
        transcriber = AssemblyAITranscriber(api_key)
    elif service == 'otherservice':
        transcriber = OtherServiceTranscriber(api_key)
    else:
        return jsonify({"error": f"Invalid transcription service: {service}"}), 400

    try:
        transcription = transcriber.transcribe_audio(audio_file)
        return jsonify({"transcription": transcription}), 200
    except Exception as e:
        logging.error(traceback.format_exc())  # Log the traceback
        return jsonify({"error": "Error during transcription"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)

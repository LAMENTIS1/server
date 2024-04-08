from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import io
import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

app = Flask(__name__)

@app.route('/')  # This is the home route
def home():
    """Provides a simple welcome message."""
    return jsonify({'message': 'Welcome to the Speech Transcription API!'})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if request.method == 'POST':
        try:
            # Securely handle file upload (avoid direct file access)
            audio_file = request.files['audio_data']
            if audio_file and allowed_audio_format(audio_file.filename):
                # Use in-memory audio processing for security
                audio_data = audio_file.read()

                # Load the model for transcription
                model_folder_path = "faster-whisper-tiny"
                model = WhisperModel(model_folder_path, device="cpu", compute_type="int8")

                # Transcribe audio
                segments, info = model.transcribe(io.BytesIO(audio_data), beam_size=5)
                transcription_result = " ".join([segment.text for segment in segments])

                # Pass the transcription result as input to GVP Bot
                response = falcon_chain.run(transcription_result)
                bot_response = response.split("GVP Bot:", 1)[-1].strip()
                start_index = bot_response.find(transcription_result) + len(transcription_result)
                response_text = bot_response[start_index:].strip()
                bot =  response_text.replace("User", "")

    # Print user input and extracted response text
                

                #return jsonify({'transcription': transcription_result , 'bot_response': bot_response  })
                return jsonify({ 'gvp bot': bot })
            else:
                return jsonify({'error': 'Invalid audio format'}), 400

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Unsupported method'}), 405

def allowed_audio_format(filename):
    """Checks if the uploaded file has a supported audio format."""
    allowed_extensions = {'wav', 'flac', 'ogg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    # Initialize GVP Bot
    os.environ['API_KEY'] = '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'
    model_id = 'tiiuae/falcon-7b-instruct'

    falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['API_KEY'],
                                repo_id=model_id,
                                model_kwargs={"temperature": 0.8, "max_new_tokens": 2000})

    template = """
    Hello! I am GVP Bot, your robotic AI assistant ready to provide helpful answers and solutions to your queries.
    User: {question}
    """

    prompt = PromptTemplate(template=template, input_variables=['question'])
    falcon_chain = LLMChain(llm=falcon_llm, prompt=prompt, verbose=True)

    #print("GVP Bot: Hello! How can I assist you today?")

    # Start Flask app
    app.run()  # Bind to all interfaces for cloud deployment

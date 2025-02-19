from flask import Blueprint, request, jsonify, render_template
from src.app_agent import process_with_llm_agent, LLMAgentOutput  # Adjust import based on your actual function
from dataclasses import asdict

# Initialize the Blueprint
routes = Blueprint('routes', __name__)

@routes.route('/')
def home():
    # Render the main HTML page (index.html)
    return render_template('index.html')

@routes.route('/process_prompt', methods=['POST'])
def process_prompt():
    # Parse JSON data from the request
    data = request.get_json()
    model = data.get('model')
    api_key = data.get('apiKey')
    prompt = data.get('prompt')
    num_tries = data.get('numTries', 10)  # Default to 1 if not provided
    
    # Call the LLM agent to process the prompt
    try:
        output: LLMAgentOutput = process_with_llm_agent(model=model, api_key=api_key, prompt=prompt, num_tries=num_tries)
        return jsonify(asdict(output))
    except Exception as e:
        # Handle any errors that occur during processing
        return jsonify({'error': str(e)}), 500

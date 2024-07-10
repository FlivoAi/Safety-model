import os
import json
import ssl
import numpy as np
import urllib.request
from flask import Flask, request, jsonify, render_template
from PIL import Image
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to allow self-signed HTTPS
def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True)

# Function to classify an image using Roboflow API
def classify_with_roboflow(image_data):
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="kTvcs84zAmxn8aoKdpZ1"
    )

    # Save the image file temporarily
    temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
    image_data.save(temp_image_path)

    result = CLIENT.infer(temp_image_path, model_id="smoke100-uwe4t/5")

    # Remove the temporary image file
    os.remove(temp_image_path)
    return result

# Updated preprocess_image function using PIL directly
def preprocess_image(image_data, target_size=(416, 416)):
    image = Image.open(image_data)
    image = image.convert('RGB')  # Ensure image is RGB (remove alpha channel if present)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize the image (same as during training if normalization was applied)
    return image_array.tolist()

# Function to send the preprocessed image to Azure for classification
def classify_with_azure(image_data):
    data = {'data': preprocess_image(image_data)}
    body = json.dumps(data).encode('utf-8')
    url = 'http://e29a2662-1955-4a29-a03a-2748dffd5ef8.centralindia.azurecontainer.io/score'
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(url, body, headers)
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf-8", 'ignore'))
        return {'error': str(error), 'details': error.read().decode("utf-8", 'ignore')}
    except json.JSONDecodeError as json_error:
        print("JSON decode error: " + str(json_error))
        return {'error': 'JSONDecodeError', 'details': str(json_error)}
    except Exception as e:
        print("An unexpected error occurred: " + str(e))
        return {'error': 'UnexpectedError', 'details': str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    image_file = request.files['image']

    # Classify with Roboflow
    roboflow_result = classify_with_roboflow(image_file)
    print(roboflow_result)
    predictions = roboflow_result['predictions']
    roboflow_result="no"
    for prediction in predictions:
        roboflow_result = "yes" if prediction['confidence']>=0.5 else "no"
        print(roboflow_result)
        # Classify with Azure
    azure_result = classify_with_azure(image_file)
    print(azure_result)
    a=azure_result['prediction1'][0][0]
    azure_result="yes" if a==1 else "no"

    return jsonify({
        'roboflow_result': roboflow_result,
        'azure_result': azure_result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
    #app.run(debug=True)
    #app.run(host='0.0.0.0', port=80)
                                                                                                                                                         66,4          47%

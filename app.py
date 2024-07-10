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

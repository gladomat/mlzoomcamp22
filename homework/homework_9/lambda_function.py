import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size=(150, 150))

# load the model
interpreter = tflite.Interpreter(model_path="dino-vs-dragon-v2.tflite")
# load the weights (in keras this is automatically)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

url = "https://upload.wikimedia.org/wikipedia/en/e/e9/GodzillaEncounterModel.jpg"

classes = ["dino", "dragon"]

def predict(url):
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_preds = preds[0].tolist()
    return float_preds

def lambda_handler(event, context):
    return predict(event['url'])
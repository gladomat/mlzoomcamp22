import bentoml

from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:latest")

# Use runner to access the model
model_runner = model_ref.to_runner()

# Use the model runner to make prediction
svc = bentoml.Service("mlzoomcamp_homework", runners=[model_runner])

# Create the endpoint service.
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(data):
    return model_runner.predict.run(data)
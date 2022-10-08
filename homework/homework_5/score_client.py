import pickle


with open('model1.bin', 'rb') as f_in:
   model = pickle.load(f_in)

with open('dv.bin', 'rb') as f_in:
   dv = pickle.load(f_in)

client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

X = dv.transform(client)
y_pred = model.predict_proba(X)[0, 1]
print('prediction', round(y_pred, 3))
print('difference', round(y_pred - 0.148, 3))
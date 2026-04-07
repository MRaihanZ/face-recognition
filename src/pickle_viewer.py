import pickle

with open("authorized_face.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data)
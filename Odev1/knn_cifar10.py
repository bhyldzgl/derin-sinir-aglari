import numpy as np
import pickle

# CIFAR-10 sınıf isimleri
classes = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

# CIFAR-10 batch dosyasını açma
with open("data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")

X_train = batch[b"data"]
y_train = batch[b"labels"]

X_train = X_train.astype(float)

print("Mesafe Türünü Seçin")
print("1 - L1 (Manhattan)")
print("2 - L2 (Euclidean)")

distance_type = input("Seçim: ")

k = int(input("k değeri girin: "))

# test görüntüsü (örnek olarak ilk görüntü)
test_image = X_train[0]
true_label = y_train[0]

distances = []

# tüm train verileri ile mesafe hesapla
for i in range(len(X_train)):

    train_image = X_train[i]

    if distance_type == "1":
        # L1 distance
        dist = np.sum(np.abs(test_image - train_image))

    else:
        # L2 distance
        dist = np.sqrt(np.sum((test_image - train_image)**2))

    distances.append((dist, y_train[i]))

# mesafeye göre sırala
distances.sort(key=lambda x: x[0])

# en yakın k komşu
neighbors = distances[:k]

labels = []

for n in neighbors:
    labels.append(n[1])

# en çok tekrar eden sınıf
prediction = max(set(labels), key=labels.count)

print("Gerçek sınıf:", classes[true_label])
print("Tahmin edilen sınıf:", classes[prediction])
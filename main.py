import tkinter as tk
from tkinter import ttk
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb

max_features = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_test, y_test))
model.save("sentiment_model.h5")

tokenizer = Tokenizer(num_words=max_features)
word_index = imdb.get_word_index()
reverse_word_index = {v + 3: k for k, v in word_index.items()}
reverse_word_index[0], reverse_word_index[1], reverse_word_index[2], reverse_word_index[3] = "<PAD>", "<START>", "<UNK>", "<UNUSED>"

def encode_review(text):
    tokens = [1]
    for word in text.lower().split():
        tokens.append(word_index.get(word, 2))
    return pad_sequences([tokens], maxlen=max_len)

model = load_model("sentiment_model.h5")

root = tk.Tk()
root.title("Movie Review Sentiment Analyzer")
root.geometry("600x400")
root.configure(bg="#1E1E1E")

frame = tk.Frame(root, bg="#1E1E1E")
frame.pack(pady=40)

title = tk.Label(frame, text="🎬 Movie Review Sentiment Analyzer", fg="#00FFAB", bg="#1E1E1E", font=("Helvetica", 18, "bold"))
title.pack(pady=10)

entry_label = tk.Label(frame, text="Enter your movie review:", fg="white", bg="#1E1E1E", font=("Helvetica", 12))
entry_label.pack(pady=5)

review_text = tk.Text(frame, height=6, width=50, bg="#2D2D2D", fg="white", insertbackground="white", font=("Helvetica", 11))
review_text.pack(pady=10)

def analyze_sentiment():
    text = review_text.get("1.0", tk.END).strip()
    if not text:
        result_label.config(text="Please enter a review.", fg="red")
        return
    encoded = encode_review(text)
    prediction = model.predict(encoded)[0][0]
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
    color = "#00FFAB" if prediction > 0.5 else "#FF6F61"
    result_label.config(text=f"Sentiment: {sentiment}", fg=color)

analyze_btn = ttk.Button(frame, text="Analyze", command=analyze_sentiment)
analyze_btn.pack(pady=10)

result_label = tk.Label(frame, text="", fg="white", bg="#1E1E1E", font=("Helvetica", 14, "bold"))
result_label.pack(pady=10)

root.mainloop()

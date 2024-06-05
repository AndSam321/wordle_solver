import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import ttk
import random

# Function to read the word list from a file efficiently
def load_word_list(filename):
    try:
        with open(filename, 'r') as file:
            words = file.read().splitlines()
        return words
    except Exception as error:
        print(f"Error reading file: {str(error)}")
        return []

# Load word list from file
word_list = load_word_list("valid-wordle-words.txt")

# Use a subset of 100 words for training
word_list = random.sample(word_list, 100)

# Function to generate feedback
def get_feedback_score(guess, correct_word):
    feedback = ['']*5
    for i in range(5):
        if guess[i] == correct_word[i]:
            feedback[i] = 'G'  # Green
        elif guess[i] in correct_word:
            feedback[i] = 'Y'  # Yellow
        else:
            feedback[i] = 'B'  # Black
    return feedback

# Create dataset efficiently
data = []

for correct_word in word_list:
    for guess in word_list:
        feedback = get_feedback_score(guess, correct_word)
        data.append((guess, feedback, correct_word))

# Convert data to DataFrame
df = pd.DataFrame(data, columns=['guess', 'feedback', 'correct_word'])

# Feature Engineering
def encode_word(word):
    return [ord(char) - ord('a') for char in word]

def encode_feedback(feedback):
    mapping = {'G': 2, 'Y': 1, 'B': 0}
    return [mapping[char] for char in feedback]
# Apply the encoding functions to the DataFrame
df['guess_encoded'] = df['guess'].apply(encode_word)
df['feedback_encoded'] = df['feedback'].apply(encode_feedback)
df['correct_word_encoded'] = df['correct_word'].apply(encode_word)

# Turn the Guess and Feedback scores into NumPy arrays for the model
X_guess = np.array(df['guess_encoded'].tolist())
X_feedback = np.array(df['feedback_encoded'].tolist())

X = np.concatenate((X_guess, X_feedback), axis=1)
y = np.array(df['correct_word_encoded'].tolist())

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape y for the classifier
y_train_reshaped = y_train.reshape((y_train.shape[0], -1))
y_test_reshaped = y_test.reshape((y_test.shape[0], -1))

# Train RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train_reshaped)

# Update the GUI 
def update_gui():
    global attempts
    global guess
    global feedback


    if guess == correct_word or attempts >= 6:
        result_label.config(text=f"Solved in {attempts + 1} attempts!" if guess == correct_word else "Failed to solve.")
        return

    if attempts == 0:
        guess = np.random.choice(word_list)
    else:
        guess_encoded = encode_word(guess)
        feedback_encoded = encode_feedback(feedback)
        X_input = np.hstack((guess_encoded, feedback_encoded)).reshape(1, -1)
        guess_encoded = model.predict(X_input).flatten()
        guess = ''.join([chr(char + ord('a')) for char in guess_encoded])

    feedback = get_feedback_score(guess, correct_word)

    # Update the grid
    for i in range(5):
        boxes[attempts][i].config(text=guess[i])
        if feedback[i] == 'G':
            boxes[attempts][i].config(bg='green', fg='white')
        elif feedback[i] == 'Y':
            boxes[attempts][i].config(bg='yellow', fg='black')
        else:
            boxes[attempts][i].config(bg='gray', fg='white')

    attempts += 1
    root.after(2000, update_gui)

# Start the game
def start_game():
    global correct_word
    global attempts
    global guess
    global feedback

    correct_word = np.random.choice(word_list)
    attempts = 0
    guess = ""
    feedback = ""
    
    # Clearing the game
    result_label.config(text="")
    for row in boxes:
        for box in row:
            box.config(text="", bg='white', fg='black')
    
    update_gui()

# Create the game
root = tk.Tk()
root.title("Wordle Solver")

# Create a grid for the guesses
boxes = []
for i in range(6):
    row = []
    for j in range(5):
        box = tk.Label(root, text="", width=4, height=2, font=("Helvetica", 24), borderwidth=2, relief="solid")
        box.grid(row=i, column=j, padx=5, pady=5)
        row.append(box)
    boxes.append(row)

# Label to display results
result_label = ttk.Label(root, text="")
result_label.grid(row=6, column=0, columnspan=5, pady=10)


# Start button
start_button = ttk.Button(root, text="Start Game", command=start_game)
start_button.grid(row=7, column=0, columnspan=5, pady=10)

# Initialize global variables
correct_word = ""
attempts = 0
guess = ""
feedback = ""

# Run the application
root.mainloop()
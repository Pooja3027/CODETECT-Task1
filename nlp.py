import subprocess
import sys
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import ttk

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import pandas as pd
except ImportError:
    install('pandas')
    import pandas as pd

try:
    import nltk
except ImportError:
    install('nltk')
    import nltk

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
except ImportError:
    install('scikit-learn')
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

# Read the local dataset
df = pd.read_csv(r'C:\Users\SMILE\task1\IMDB Dataset.csv')

# Assuming the CSV has columns 'review' and 'sentiment'
reviews = df['review'].values
sentiments = df['sentiment'].values

# Calculate counts of positive and negative reviews
positive_count = sum(sentiment == 'positive' for sentiment in sentiments)
negative_count = sum(sentiment == 'negative' for sentiment in sentiments)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)

# Create and train the model
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(X_train, y_train)

# Evaluate the model
predictions = text_clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

def show_results_in_new_window():
    result_text = (
        f'Number of Positive Reviews: {positive_count}\n'
        f'Number of Negative Reviews: {negative_count}\n\n'
        f'Accuracy: {accuracy:.2f}\n\n'
        f'Classification Report:\n{report}'
    )
    result_window = tk.Toplevel(root)
    result_window.title("NLP Sentiment Analysis Results")
    result_window.geometry("800x600")
    
    # Create a Text widget for the results with proper configuration
    result_text_widget = tk.Text(result_window, wrap='word', font=("Courier New", 20), bg='#f0f8ff', padx=10, pady=10)
    result_text_widget.pack(fill='both', expand=True)

    # Insert formatted results and configure the widget
    result_text_widget.insert(tk.END, result_text)
    result_text_widget.config(state=tk.DISABLED)  # Make the text widget read-only

# Create the main GUI application
root = tk.Tk()
root.title("NLP Sentiment Analysis")
root.geometry("400x300")

# Button to trigger the result display in a new window
show_button = ttk.Button(root, text="Show Results", command=show_results_in_new_window)
show_button.pack(pady=20)

# Run the application
root.mainloop()

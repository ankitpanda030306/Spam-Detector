##📬 Spam Message Detector (Machine Learning Project)

A beginner-friendly machine learning project to classify SMS messages as **spam** or **not spam** using text preprocessing, TF-IDF vectorization, and Naive Bayes classification.

> 🧠 Built during my 1st year of B.Tech while learning AI  
> ✅ Guided by Google's ML Crash Course and Colab hands-on labs

---

##🚀 Features
- Cleaned and preprocessed real-world SMS dataset
- TF-IDF vectorization to convert text into numerical data
- Used `MultinomialNB` model for spam classification
- Evaluated model using accuracy, confusion matrix, and ROC curve
- Saved the trained model and vectorizer using `joblib`
- Easy interface to test custom SMS input

---

##🛠️ Technologies Used
- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Google Colab

---

##🧪 Try It Yourself
```python
predict_spam("You have won ₹1,00,000! Click here to claim.")
# Output: Spam

##📁 Project Structure
spam-detector/
├── Spam_detector.ipynb        # Main notebook
├── spam_detector_model.pkl    # Trained model file
├── vectorizer.pkl             # TF-IDF vectorizer
└── README.md                  # Project description

##🙌 Acknowledgements
- Google Machine Learning
- UCI SMS Spam Dataset

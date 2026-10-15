from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ["win money now", "hello how are you", "free prize click", "meet tomorrow"]
labels = [1, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

message = input("Message: ")
prediction = model.predict(vectorizer.transform([message]))[0]

print("SPAM" if prediction == 1 else "NORMAL")

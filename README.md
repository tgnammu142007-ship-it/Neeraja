PocketSmart AI: Your Smart Budget & Recommendation Assistant
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
CORS(app)

# Dataset for expense categorization
training_data = [
    ("Walmart grocery store", "Groceries"),
    ("Uber ride downtown", "Transportation"),
    ("Netflix monthly subscription", "Entertainment"),
    ("Starbucks coffee", "Food & Dining"),
    ("Electricity bill payment", "Utilities"),
    ("Amazon electronics purchase", "Shopping"),
    ("Pharmacy prescription medicine", "Health & Wellness")
]

X_train, y_train = zip(*training_data)

# Machine Learning Pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

@app.route('/api/categorize', methods=['POST'])
def categorize():
    data = request.json
    description = data.get('description', '')
    if not description:
        return jsonify({'error': 'No description provided'}), 400
    
    predicted_category = model.predict([description])[0]
    
    # Generate mock AI recommendation based on category
    recommendations = {
        "Food & Dining": "Dining out accounts for higher spending. Consider meal prepping to save ~$40 this week.",
        "Entertainment": "Review active recurring subscriptions to cut unused digital streaming services.",
        "Groceries": "Buying in bulk at wholesale outlets can lower weekly grocery bills by 15%.",
        "Transportation": "Combine daily errands or consider public transit to minimize fuel expenses."
    }
    
    rec = recommendations.get(predicted_category, "Keep tracking expenses daily to maintain a balanced budget.")
    
    return jsonify({
        'description': description,
        'category': predicted_category,
        'recommendation': rec
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
    

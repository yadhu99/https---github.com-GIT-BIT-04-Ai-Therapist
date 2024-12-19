from flask import Flask, request, jsonify
import pickle
from demo3 import get_content_based_recommendations
from demo3 import get_collaborative_filtering_recommendations

app = Flask(__name__)


# Load the hybrid model
with open("hybrid_recommender_model.pkl", "rb") as hybrid_model_file:
    hybrid_model = pickle.load(hybrid_model_file)

algo = hybrid_model["collaborative_filtering_model"]
tfidf_vectorizer = hybrid_model["tfidf_vectorizer"]
content_df = hybrid_model["content_df"]

def get_recommendations(user_id, targeted_issue, top_n=5):
    # Generate content-based recommendations
    content_based_recommendations = get_content_based_recommendations(targeted_issue, top_n)
    
    # Generate collaborative filtering recommendations
    collaborative_filtering_recommendations = get_collaborative_filtering_recommendations(user_id, top_n)
    
    # Combine them into hybrid recommendations
    hybrid_recommendations = list(set(content_based_recommendations + collaborative_filtering_recommendations))
    return hybrid_recommendations[:top_n]

@app.route('/get_recommendations', methods=['POST'])
def recommend():
    user_data = request.get_json()
    user_id = user_data['user_id']
    targeted_issue = user_data['targeted_issue']
    
    recommendations = get_recommendations(user_id, targeted_issue)
    
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run( host='localhost', port=8000 ,debug=True)

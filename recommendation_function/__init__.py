
import azure.functions as func
import joblib
import pandas as pd
import os
import json
import tempfile
import urllib.request
import logging


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

cache = {}
model_url = 'https://ocstoragendv.blob.core.windows.net/fichiers/svd.joblib?sp=r&st=2025-04-14T16:25:51Z&se=2025-04-28T23:25:51Z&sv=2024-11-04&sr=b&sig=XrYR4FJjPzxZheHrCCE%2BSnbW%2FeCO3YEoR569%2BBpHzn0%3D'
csv_url = "https://ocstoragendv.blob.core.windows.net/fichiers/user_article_interaction_scaled.csv?sp=r&st=2025-04-14T16:28:47Z&se=2025-04-21T23:28:47Z&sv=2024-11-04&sr=b&sig=PHCjP7EAfZnF4K%2FSw7PrMdU%2B3aeYw3nJ96aii03%2B%2FlQ%3D"

def download_once(url, filename):
    local_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)
    return local_path

@app.function_name(name="recommendation_function")
@app.route(route="recommend", methods=["GET"])
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Recommendation function received a request.')

    try:
        user_id = req.params.get('user_id')
        if not user_id:
            return func.HttpResponse("Please provide user_id in the query string.", status_code=400)
        user_id = int(user_id)
        
        if 'loaded_model' not in cache or 'user_data' not in cache:
            logging.info("Loading model and data...")
            model_path = download_once(model_url, 'svd.joblib')
            csv_path = download_once(csv_url, 'user_article_interaction_scaled.csv')

            logging.info(model_url)
            cache['loaded_model'] = joblib.load(model_path)
            logging.info("✅ Modèle chargé")
            df = pd.read_csv(csv_path)
            logging.info("✅ Données chargées")
            cache['user_data'] = df
            cache['all_articles'] = df['article_id'].unique()

            
            logging.info("✅ Modèle et données chargés en cache")

        loaded_model = cache['loaded_model']
        df = cache['user_data']
        all_articles = cache['all_articles']

        articles_seen = df[df['user_id'] == user_id]['article_id'].unique()
        unseen_articles = [aid for aid in all_articles if aid not in articles_seen]

        predictions = []
        for article_id in unseen_articles:
            try:
                logging.info(f"Predicting for user {user_id}, article {article_id}")
                est_rating = loaded_model.predict(user_id, article_id).est
                predictions.append((article_id, est_rating))
            except Exception as e:
                logging.warning(f"Prediction failed for user {user_id}, article {article_id}: {e}")

        top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
        recommendations = [{"article_id": str(aid), "rating": round(score, 2)} for aid, score in top_predictions]
        logging.info(func.HttpResponse(
            json.dumps({"recommendations": recommendations}),
            mimetype="application/json"
            ))
        return func.HttpResponse(
            json.dumps({"recommendations": recommendations}),
            mimetype="application/json"
            )

    except Exception as e:
        logging.error(f"Error during recommendation: {e}")
        return func.HttpResponse(
            f"Error: {str(e)}", status_code=500
            )

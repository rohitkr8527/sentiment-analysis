FROM python:3.10-slim

WORKDIR /app

COPY fastapi_app/ /app/

COPY models/tfidf_vectorizer.pkl /app/models/tfidf_vectorizer.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

#local
# CMD ["python", "app.py"]  

#Prod
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]

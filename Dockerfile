# python base image (the same version 3.12 that was used to train model.joblib)
FROM python:3.12-slim

# working directory inside the container
WORKDIR /app

# install dependencies first, so Docker can cache this step
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the project files (main.py, model.joblib and others)
COPY . .

EXPOSE 8000

# run the FastAPI app with uvicorn, 0.0.0.0 so it is reachable from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

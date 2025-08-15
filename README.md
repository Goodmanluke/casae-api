# Casae API

This is the FastAPI backend for the Casae application.

## Development

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the application locally:

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

## Docker

Build and run the API with Docker:

```bash
docker build -t casae-api .
docker run -p 8000:8000 casae-api
```

# redeploy trigger
# redeploy trigger again


# redeploy trigger third time

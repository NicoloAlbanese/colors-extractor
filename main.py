from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router as api_router


# Create a FastAPI application instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,            # Add Cross-Origin Resource Sharing (CORS) middleware to handle browser security restrictions
    allow_origins = ["*"],     # Allow requests from all origins (insecure, for development only)
    allow_credentials = True,  # Allow credentials like cookies in the requests
    allow_methods = ["*"],     # Allow all HTTP methods (GET, POST, etc.)
    allow_headers = ["*"],     # Allow all headers in requests
)

# Define a GET request handler for the root endpoint ("/")
@app.get("/")
# Define an asynchronous function for the root endpoint
async def root():
    # Return a JSON response with a message
    return {"message": "API for color extraction from images."}

# Include the API router with a prefix of "/api"
app.include_router(api_router, prefix="/api")
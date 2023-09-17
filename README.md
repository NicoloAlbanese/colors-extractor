# colors-extractor

![Colors Extractor](/img/front.png "Colors Extractor")

### Description

REST Service to extract predominant colors from input images. Tools:

- Unsupervised Machine Learning techniques (K-Means) to extract predominant colors from an image.
- FastAPI to serve the solution as a REST API.
- Docker for isolated and consistent deployment.

### Usage

To create a Docker image and run a container from command line:

```
cd colors-extractor

docker build -t colors-extractor .

# Replace port number with desired port
docker run -d -p 8000:8000 colors-extractor
```

To test the API service:

```
curl --location 'http://<host>:<port>/api/colors' \
--header 'Content-Type: application/json' \
--data '{
    "url_or_path": "https://fastly.picsum.photos/id/63/5000/2813.jpg?hmac=HvaeSK6WT-G9bYF_CyB2m1ARQirL8UMnygdU9W6PDvM",
    "num_clusters": 3
}'
```

API documentation available at:

```
http://<host>:<port>/docs
```

### Author

- Nicolò Cosimo Albanese, nicolo_albanese@outlook.it

### License

- MIT License

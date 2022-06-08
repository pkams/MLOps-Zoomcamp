## Deploying a model as web-service

* Creating virtual environment with pipenv
* Creating a script for predicting
* Putting the script into a Flask app
* Packaging the app to docker

```bash
docker build -t ride-duration-prediction-service:v1 .```
```

```bash
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```
--rm remove image after 

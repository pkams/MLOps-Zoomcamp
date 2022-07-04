FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim
# prepared docker image

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["main.py", "./"]

ENTRYPOINT ["python", "main.py"]


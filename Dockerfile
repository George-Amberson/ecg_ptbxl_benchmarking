FROM ubuntu:22.04
FROM python:3.10

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
WORKDIR /app
VOLUME /app_volume
COPY . /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install -r requirements.txt
ADD https://dropmefiles.com/3NLLQ /
CMD [ "./get_datasets.sh" ]
CMD ["python", "./code/test.py"]
CMD ["python", "./code/test.py"]
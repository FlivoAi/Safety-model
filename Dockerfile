# Use the official Python base image (choose an appropriate version)
FROM python:3.9

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install app dependencies
RUN pip install -r requirements.txt

# Bundle your app source code into the container
COPY . .

EXPOSE 80

CMD [ "python3", "app.py","--host","0.0.0.0","--port","80"]

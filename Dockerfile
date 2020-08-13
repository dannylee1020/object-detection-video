FROM python:3.7

# define working directory within docker image
WORKDIR /opt/object_detection_video

# copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt


# copy necessary folders to run the app
COPY app /opt/object_detection_video/app/
COPY model /opt/object_detection_video/model/
COPY output /opt/object_detection_video/output/
COPY videos /opt/object_detection_video/videos/


# for local build
EXPOSE 8501


# for running locally
ENTRYPOINT ["streamlit", "run"]
CMD ["app/app.py"]


FROM python:3.9-slim

# don't create .pyc files on the import of source files
ENV PYTHONDONTWRITEBYTECODE = 1
# see input/output in real time in terminal
ENV PYTHONUNBUFFERED = 1

#set working directory (where COPY– RUN– EXPOSE– is run from)
WORKDIR /app

# install requirements.txt (generated that using pipreqs) using python -m syntax, because I wanna install the packages into the current working python version
COPY requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

COPY main.py .

EXPOSE 800

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



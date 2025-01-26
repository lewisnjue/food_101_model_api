FROM python:3.10-slim

WORKDIR /food

COPY . /food/

RUN chmod +x install.sh && \
    ./install.sh

CMD ["python", "./src/main.py"]

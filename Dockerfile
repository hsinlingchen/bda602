FROM python:3.9.6
# --platform-linux/amd64
ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*


# Get necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt


COPY ./src/hw5/hw5.sql .
COPY ./src/hw5/hw5.py .
COPY ./src/hw5/mid_analyzer.py .
COPY ./src/hw5/diff_w_mean.py .
COPY ./src/hw5/plot.py .
COPY ./src/final.sh .

CMD ./final.sh


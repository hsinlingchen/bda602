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


COPY ./src/hw6/hw6.sql .
COPY ./src/hw6/check_database.sh .
COPY ./baseball.sql .

RUN chmod +x ./check_database.sh
CMD ./check_database.sh


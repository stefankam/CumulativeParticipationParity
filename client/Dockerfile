FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# --------------------------
# 1. Install system packages
# --------------------------
RUN apt-get update && apt-get install -y \
    git curl sudo python3 python3-pip python3-dev \
    openvswitch-switch \
    net-tools iproute2 iputils-ping \
    default-jre default-jdk \
    build-essential \
    lsb-release software-properties-common \
    && rm -rf /var/lib/apt/lists/*


# --------------------------
# 3. Install Python packages
# --------------------------
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# --------------------------
# 4. Set working directory
# --------------------------
WORKDIR /app

# Copy main Python scripts
COPY main_client.py .
COPY topology_client.py .
COPY availability.py .

COPY traces/ traces/

RUN apt-get update && apt-get install -y iputils-ping

# --------------------------
# 5. Set environment + entry
# --------------------------
ENV DEVICE_ID=h0
ENV FLASK_PORT=5000

EXPOSE 5000

CMD ["python3", "-u", "main_client.py"]

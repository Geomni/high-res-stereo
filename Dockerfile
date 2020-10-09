FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
COPY ./ /workspace
RUN pip install -r requirements.txt
RUN apt-get update -y && apt-get install -y wget unzip && apt-get clean -y
RUN wget "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" && \
    unzip awscli-exe-linux-x86_64.zip && \
    sh aws/install && \
    rm -rf aws awscli-exe-linux-x86_64.zip
    

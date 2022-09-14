FROM onnxruntime-trt:cuda10.1
USER root
RUN apt-get update -y && \
    apt-get install -y \
    vim libgl1-mesa-glx curl libglib2.0-0\
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/li \
    apt install build-essentia
RUN conda install -c anaconda python=3.7.11 -y


COPY ../requirements.txt ./
ENV PATH="/root/.local/bin:${PATH}"
RUN pip install --upgrade pip --ignore-installed
RUN pip install --no-cache-dir --ignore-installed --default-timeout=100 -r requirements.txt
RUN pip install --ignore-installed -U scikit-image==0.19.1 \
    && pip install --ignore-installed -U albumentations==1.1.0
RUN pip install --ignore-installed uvloop
RUN conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch -y



EXPOSE 8080
ENTRYPOINT ["/bin/bash", "start.sh"]

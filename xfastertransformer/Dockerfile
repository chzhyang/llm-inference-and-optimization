ARG UBUNTU_VER=22.04
FROM ubuntu:${UBUNTU_VER} as demo

ENV LANG C.UTF-8
# ENV http_proxy http://proxy-prc.intel.com:911
# ENV https_proxy http://proxy-prc.intel.com:912
# ENV no_proxy localhost,127.0.0.1,*.intel.com

SHELL ["/bin/bash", "--login", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    autoconf \
    build-essential \
    gcc-12 \
    g++-12 \
    make \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libnuma-dev \
    numactl \
    time \
    git \
    wget \
    pkg-config \
    software-properties-common \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 60 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/oneapi-src/oneCCL.git /tmp/oneCCL
WORKDIR /tmp/oneCCL
RUN git checkout 2021.10 \
    && sed -i 's/cpu_gpu_dpcpp/./g' cmake/templates/oneCCLConfig.cmake.in \
    && mkdir build
WORKDIR /tmp/oneCCL/build
RUN cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/oneCCL \
    && make -j install 
RUN echo "source /usr/local/oneCCL/env/setvars.sh" >> ~/.bashrc
WORKDIR /usr/local/
RUN rm -rf /tmp/oneCCL

# RUN wget -q https://github.com/oneapi-src/oneDNN/releases/download/v0.21/mklml_lnx_2019.0.5.20190502.tgz \
#     && tar -xzf mklml_lnx_2019.0.5.20190502.tgz \
#     && rm -f mklml_lnx_2019.0.5.20190502.tgz
COPY mklml_lnx_2019.0.5.20190502.tgz .
RUN tar -xzf mklml_lnx_2019.0.5.20190502.tgz \
    && rm -f mklml_lnx_2019.0.5.20190502.tgz
RUN echo "export LD_LIBRARY_PATH=/usr/local/mklml_lnx_2019.0.5.20190502/lib:\$LD_LIBRARY_PATH" >> /root/.bashrc

RUN apt-get remove -y \
    git \
    wget \
    libbz2-dev

ENV PATH /opt/miniconda3/bin:$PATH

WORKDIR /app

# COPY api.py adapter.py model_worker.py utils.py llm_metrics.py ./
COPY automodel.py generate.py ./

ENV KMP_BLOCKTIME=1
ENV KMP_SETTINGS=1
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV OMP_NUM_THREADS=30

CMD ["conda", "run", "--no-capture-output", "-n", "llmenv", "python", "api.py"]

# FROM tensorflow/tensorflow:2.18.0rc0-gpu-jupyter
# FROM tensorflow/tensorflow:2.17.0-gpu-jupyter
# FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install basic utilities and Python
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    gnupg2 \
    emacs \
    less \
    python3 \
    python3-pip \
    python-is-python3

# Install Node.js and Yarn
RUN curl -sL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - && \
    echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y yarn


# Create a non-root user with sudo privileges
ARG USERNAME=dockeruser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to the non-root user
USER $USERNAME
WORKDIR /app
RUN chown -R $USERNAME:$USERNAME /app
ENV PYTHONPATH="/app"

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install JAX with CUDA 12.6 support
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install playwright for frontend testing
RUN pip install playwright
RUN python -m playwright install
RUN python -m playwright install-deps
RUN python -m playwright install chromium

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install TypeScript and related tools
RUN yarn add typescript@5.5 --dev
RUN yarn add eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin --dev
RUN yarn add prettier --dev
RUN yarn add husky lint-staged --dev

# Install Bootstrap type definitions
RUN yarn add @types/bootstrap --dev

COPY rgi rgi
COPY scripts scripts
COPY notebooks notebooks

# Update .bashrc to source the custom rgi.bashrc file
RUN echo "source /app/scripts/rgi.bashrc" >> /home/$USERNAME/.bashrc

CMD ["bash"]

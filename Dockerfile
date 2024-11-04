FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

# Install basic utilities and Python
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    gnupg2 \
    emacs \
    less \
    python3 \
    python3-pip \
    python-is-python3 \
    python3-venv \
    git

# Install Node.js and Yarn
RUN curl -sL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - && \
    echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y yarn

ARG USERNAME=ubuntu
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chmod 0440 /etc/sudoers

# Switch to the non-root user
USER $USERNAME
WORKDIR /app
RUN chown -R $USERNAME:$USERNAME /app

# Install TypeScript and related tools
RUN yarn add typescript@5.5 --dev
RUN yarn add eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin --dev
RUN yarn add prettier --dev
RUN yarn add husky lint-staged --dev

# Install Bootstrap type definitions
RUN yarn add @types/bootstrap --dev

# Create a virtual environment
ENV PYTHONPATH="/app"
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Upgrade pip
RUN python3 -m pip install --upgrade pip


# Install playwright & chromium for frontend testing
# These are slow to install, so adding them here makes requirements.txt installs faster.
# ## NOTE: Updates here must also be changed in requirements.in
RUN pip install playwright==1.48.0
RUN python -m playwright install-deps
RUN python -m playwright install chromium

# Install JAX, torch & tensorflow before requirements.txt is processed.
# These are slow to install, so adding them here makes requirements.txt installs faster.
# ## NOTE: Updates here must also be changed in requirements.in
RUN pip install --upgrade "jax[cuda12]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install tensorflow==2.18.0

RUN pip install pip-tools
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts scripts

# Update .bashrc to source the custom rgi.bashrc file
RUN echo "source /app/scripts/rgi.bashrc" >> /home/$USERNAME/.bashrc

CMD ["bash"]

FROM tensorflow/tensorflow:2.17.0-gpu-jupyter

# Create a non-root user with sudo privileges
ARG USERNAME=dockeruser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update && apt-get install -y sudo
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /app

# Install Node.js and Yarn
RUN apt-get update && apt-get install -y curl gnupg2 && \
    curl -sL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - && \
    echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y yarn

# Check versions
RUN node -v && yarn -v

# Install playwright for frontend testing
RUN pip install playwright
RUN python -m playwright install
RUN python -m playwright install-deps
RUN python -m playwright install chromium

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install useful tools
RUN apt-get update && apt-get install -y emacs less

# Install TypeScript and related tools
RUN yarn add typescript@5.5 --dev
RUN yarn add eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin --dev
RUN yarn add prettier --dev

# Install Bootstrap type definitions
RUN yarn add @types/bootstrap --dev

COPY rgi rgi
COPY scripts scripts
COPY notebooks notebooks

# Change ownership of the /app directory
RUN chown -R $USERNAME:$USERNAME /app

# Run ldconfig as root
RUN ldconfig

# Switch to the non-root user
USER $USERNAME

ENV PYTHONPATH="/app"

CMD ["bash"]

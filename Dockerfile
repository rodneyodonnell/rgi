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

# Install playwright for frontend testing
RUN pip install playwright
RUN python -m playwright install
RUN python -m playwright install-deps
RUN python -m playwright install chromium

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


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

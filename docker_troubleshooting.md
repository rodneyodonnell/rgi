# GPU Docker Troubleshooting Guide

## Quick Fix Steps (Try these first)

1. Restart the NVIDIA driver:
   ```bash
   sudo systemctl restart nvidia-persistenced
   ```

2. Unload and reload the NVIDIA kernel modules:
   ```bash
   sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
   sudo modprobe nvidia
   ```

3. Restart the Docker service:
   ```bash
   sudo systemctl restart docker
   ```

4. If using nvidia-docker2, restart the NVIDIA Docker runtime:
   ```bash
   sudo systemctl restart nvidia-docker
   ```

5. Check NVIDIA driver status:
   ```bash
   nvidia-smi
   ```

## If Quick Fix Doesn't Work

6. Stop all Docker containers, prune Docker system, and restart Docker:
   ```bash
   docker stop $(docker ps -aq)
   docker system prune -f
   sudo systemctl restart docker
   ```

7. If using Docker Desktop, restart it from the system tray or application menu.

## Suspend/Resume Fix

To address issues related to suspend/resume:

1. Create a system resume script:
   ```bash
   sudo nano /lib/systemd/system-sleep/nvidia-resume
   ```

2. Add the following content:
   ```bash
   #!/bin/bash
   case $1 in
       post)
           # Reload NVIDIA modules
           rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
           modprobe nvidia
           # Restart Docker
           systemctl restart docker
           ;;
   esac
   ```

3. Make the script executable:
   ```bash
   sudo chmod +x /lib/systemd/system-sleep/nvidia-resume
   ```

## Additional Considerations

- Update NVIDIA drivers to the latest version
- Check for CUDA toolkit updates
- Ensure Docker and NVIDIA Container Toolkit are up to date

## Conversation Starter

Use this prompt to kickstart a conversation with an AI assistant about this issue:

"I'm experiencing issues with GPU support in Docker containers on my Linux system. The last time this happened, rebooting fixed the problem, but I'd like to avoid that if possible. I've tried the steps in my GPU Docker troubleshooting guide, but I'm still having problems. The specific error I'm seeing is [INSERT YOUR SPECIFIC ERROR MESSAGE HERE]. Can you help me diagnose and resolve this issue without rebooting?"

## Docker Command Cheat Sheet

### Basic Docker Commands

1. List running containers:
   ```bash
   docker ps
   ```

2. List all containers (including stopped):
   ```bash
   docker ps -a
   ```

3. List images:
   ```bash
   docker images
   ```

### Building and Running

4. Build an image (run in the directory with the Dockerfile):
   ```bash
   docker build -t your-image-name:tag .
   ```

5. Run a container with GPU support:
   ```bash
   docker run -it --gpus all -p 8888:8888 -v /path/on/host:/path/in/container your-image-name:tag
   ```

6. Run a specific command in a new container:
   ```bash
   docker run -it --gpus all your-image-name:tag python -c "import jax; print(jax.devices())"
   ```

### Managing Containers

7. Stop a running container:
   ```bash
   docker stop container_id_or_name
   ```

8. Remove a container:
   ```bash
   docker rm container_id_or_name
   ```

9. Remove all stopped containers:
   ```bash
   docker container prune
   ```

### Managing Images

10. Remove an image:
    ```bash
    docker rmi image_id_or_name
    ```

11. Remove all unused images:
    ```bash
    docker image prune -a
    ```

### System Cleanup

12. Remove all unused containers, networks, images (both dangling and unreferenced), and optionally, volumes:
    ```bash
    docker system prune -a
    ```

### Logs and Debugging

13. View container logs:
    ```bash
    docker logs container_id_or_name
    ```

14. Execute a command in a running container:
    ```bash
    docker exec -it container_id_or_name bash
    ```

### Project-Specific Commands

15. Rebuild the project image (adjust the path as needed):
    ```bash
    docker build -t rgi-dev-container:latest /path/to/your/project
    ```

16. Run the project container with all necessary mounts and GPU support:
    ```bash
    docker run -it --gpus all \
      -v "$(pwd)/logs:/app/logs" \
      -v "$(pwd)/rgi:/app/rgi" \
      -v "$(pwd)/web_app:/app/web_app" \
      -v "$(pwd)/scripts:/app/scripts" \
      -v "$(pwd)/notebooks:/app/notebooks" \
      -p 8888:8888 \
      --name rgi-dev-container \
      rgi-dev-container:latest
    ```

17. Start Jupyter Notebook server in the container:
    ```bash
    docker exec -it rgi-dev-container jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
    ```

Remember to replace `your-image-name`, `container_id_or_name`, and other placeholders with your actual project-specific names and IDs.
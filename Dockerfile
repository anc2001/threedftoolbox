FROM pymesh/pymesh:py3.7-slim

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Build argument to pass the host user UID
ARG USER_UID=1000

# Create a user with the specified UID
RUN useradd -m -u $USER_UID -s /bin/bash user \
    && echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to the new user
USER user

# Copy the project files into the container
COPY --chown=user:user . .

RUN sudo chown -R user:user /app

# Install the Python package and its dependencies
RUN pip install .

# Set the default command to run a bash shell when the container starts
CMD ["bash"]     

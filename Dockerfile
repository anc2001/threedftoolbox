FROM pymesh/pymesh:py3.7-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies, Python 3 with pip, and C++ tools for building PyMesh and Draco
RUN apt-get update && apt-get install -y \
    sudo \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install the Python package and its dependencies
RUN pip install . 

# Set the default command to run a bash shell when the container starts
CMD ["bash"]

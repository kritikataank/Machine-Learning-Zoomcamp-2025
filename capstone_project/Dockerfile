# 1. Base Image: Use a slim version of Python for a smaller image size
FROM python:3.10-slim

# 2. Set Environment Variable: Tells Flask to listen on all interfaces
ENV FLASK_APP=predict.py

# 3. Create and Set Working Directory
# This is where the application files will live inside the container
WORKDIR /app

# 4. Copy Requirements: Copy the requirements file into the working directory
COPY requirements.txt requirements.txt

# 5. Install Dependencies: Install all necessary Python packages
# --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Files: Copy the rest of the files into the container
# This includes the prediction script and the saved model/vectorizer files
COPY . .

# 7. Expose Port: Inform Docker that the container listens on this port
# This matches the port used in your predict.py (port=9699)
EXPOSE 9699

# 8. Define Startup Command: The command to run when the container starts
# The host '0.0.0.0' allows the container to be accessed externally
CMD ["flask", "run", "--host=0.0.0.0", "--port=9699"]

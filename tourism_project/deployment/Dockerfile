# Use a minimal base image with Python 3.9 installed
FROM python:3.9

# Create a non-root user and set up its home directory
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory for the non-root user
WORKDIR /home/user/app

# Copy requirements.txt first to leverage Docker's build cache
# Assuming requirements.txt is in the build context root (i.e., tourism_project/deployment)
COPY requirements.txt .

# Install Python dependencies into the user's home directory
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files to the working directory, owned by the user
# Assuming app.py and best_model.joblib are also in the build context root
COPY . .

# Define the command to run the Streamlit app on port "8501" and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]

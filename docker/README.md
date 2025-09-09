# Docker Setup

## Build and Run

```bash
# Build image
cd dcoker 
docker build -t book:latest .

# Run container with proper file permissions
cd ..
docker run -d -p8888:8888 --init --rm -it --gpus=all --ipc=host \
  --user=$(id -u):$(id -g) \
  -e HOME=/workspace \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  --name="BookCover" --env TZ=YOUR_TIME_ZONE \
  --volume=$PWD:/workspace \
  --volume "YOUR_DATASET_PATH:/dataset:ro" \
  book:latest

# Start Jupyter Lab
docker exec -itd BookCover jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token='YOURPASS'

# Access container shell
docker exec -it BookCover bash
```

## Usage

```bash
# Training
python src/train.py

# Testing  
python src/test.py
```



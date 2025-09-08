docker build --pull --rm -f "Dockerfile" -t book:latest "."

docker run -d -p8888:8888 --init --rm -it --gpus=all --ipc=host --user=$(id -u):$(id -g) --name="BookCover" --env TZ=Asia/Tokyo --volume=$PWD:/workspace --volume "/home/disk/book_cover_classification/dataset:/dataset:ro" book:latest fish

docker exec -itd BookCover jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token='daichi'

docker exec -it BookCover fish

docker exec -it -u root BookCover fish



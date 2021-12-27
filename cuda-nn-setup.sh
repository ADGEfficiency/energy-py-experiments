sudo dpkg -i ./cudnn-local-repo-1804-8.3.1.22_1.0-1_amd64.deb
sudo apt-key add /var/cudnn-local-repo-*/7fa2af80.pub
sudo apt-get update
sudo apt-get install libcudnn8=8.3.1.22-1+cuda11.5
sudo apt-get install libcudnn8-dev=8.3.1.22-1+cuda11.5
sudo apt-get install libcudnn8-samples=8.3.1.22-1+cuda11.5
sudo apt-get install libfreeimage3 libfreeimage-dev
sudo apt-get install cuda-11-2 libcusolver-11-0
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.0/lib64

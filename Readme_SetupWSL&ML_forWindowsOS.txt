WSL runs linux commands in windows OS.
	
	1. Run wsl --install in powershell. Restart computer.
	2. Run wsl.exe  --install -d ubuntu. Create Linux user and pw.
	3. Install Windows Terminal https://learn.microsoft.com/en-us/windows/terminal/get-started

GPU acceleration:
	1. Update Nvidia drivers 
	2. Download and install docker: https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
	3. Run in bash to test ML: docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tensorflow:20.03-tf2-py3
Continue ML tutorial here: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute
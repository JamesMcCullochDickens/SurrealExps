conda create --name SurrealExps python=3.10 -y
conda activate SurrealExps
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install matplotlib
pip install seaborn
pip install pandas
pip install decord
pip install opencv-python
pip install scipy
pip install open3d
pip install ultralytics
pip install psutil
## Setup
* `git clone https://github.com/okamiRvS/BERT-BinaryLanguageClassifier.git`
* `cd BERT-BinaryLanguageClassifier`
* `docker build -t fastapiapp:latest -f docker/Dockerfile .`
* `docker run -p 5000:80 fastapiapp:latest`
* Connect to `http://localhost:5000/docs`
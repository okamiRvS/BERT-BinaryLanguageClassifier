## Setup
1. `git clone https://github.com/okamiRvS/BERT-BinaryLanguageClassifier.git`
2. `cd BERT-BinaryLanguageClassifier`
3. `mkdir models`
4. Download the [model](https://drive.google.com/file/d/1rFs4CCk7R0My-gRMbMyF4e2YW95aIVYJ/view?usp=sharing), unzip it and put the `finetuning-binary-language-classifier` folder into `models` folder
5. `docker build -t fastapiapp:latest -f docker/Dockerfile .`
6. `docker run -p 5000:80 fastapiapp:latest`
7. Connect to `http://localhost:5000/docs`
8. or make a post at this link `http://localhost:5000/predict` which takes a JSON payload like `{"text":"Questa è una frase in Italiano!"}`
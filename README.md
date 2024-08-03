
## Setting up
Create a virtual environment & activate using appropriate command for your OS
`
python -m venv <<path>>
<<path>>/Scripts/Activate.ps1
`shell
Clone the code in a directory, CD into it, and install dependencies
`
pip install -r requirements.txt
`

Below are different implementations as of date:

### HSN & GL Code Classifier 
Classify a given invoice item into a GL Code and HSN code. This is a supervised learning problem and supports 2 models:
1. SVM with TF-IDF
2. EMbeddings with LSTM
 
Copy the training and test csvs in ./data folder & train the models. The filenames are hard-coded so use the same name as code for now.

```
python main.py --train --method embedding_lstm --type GL_Code
python main.py --train --method tfidf_svm --type GL_Code
python main.py --train --method embedding_lstm --type HSN_Code
python main.py --train --method tfidf_svm --type HSN_Code

python main.py --test --method embedding_lstm --type GL_Code
python main.py --test --method tfidf_svm --type GL_Code
python main.py --test --method embedding_lstm --type HSN_Code
python main.py --test --method tfidf_svm --type HSN_Code
```

Now run the FAST API front-end `python main.py`  & navigate to http://localhost:8000/. The JS function that hooks into the API is called `predict()`



### GL Line Item Recon
Customer has a historical data of vendors' invoice items which were not paid along with the reason in a csv file. Now, given a new line item which is in suspense, find from historical data, top k similar items and show the reasons along with them. This module shares same embeddings and simiarity calcuation as SKUs Mapping module above.

Create embeddings:

`python .\main.py --embeddings <input.csv>`



`python .\main.py --find 'new line item'`

you could also run the Fast server to get this via the UI. JS method is `findSimilar()`

### Sentiment and Insights extractor
Simple function to calculate sentiment for all rows in a csv file and extract key insights.

`python sentiment_analysis.py <input_file> <output_file>` 

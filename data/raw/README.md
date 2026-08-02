# Raw data

Download the dataset from Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud

Place the file here as:

```
data/raw/creditcard.csv
```

It's not committed to the repo and is gitignored; at 144MB it exceeds GitHub's 100MB per-file limit, and shipping raw data
through git is bad practice regardless. The deployed dashboard doesn't need
this file (or most of `data/`/`models/`) at all — see the "Deploying" section
in the main README.

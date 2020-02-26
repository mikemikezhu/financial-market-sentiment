 # Install nbconvert package
 pip install --upgrade nbconvert

 # Remove previously generated README.md file
 rm -rf market_sentiment/
 rm -rf README.md

 # Convert jupyter notebook to markdown
 jupyter nbconvert --to markdown market_sentiment.ipynb

 # Rename README.md
 mv market_sentiment.md README.md

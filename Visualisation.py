# -*- coding: latin-1 -*-

import pandas as pd
import numpy as np
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import warnings

dataFiles = ['tweet-2009', 'tweet-2010', 'tweet-2011', 'tweet-2012', 'tweet-2013', 'tweet-2014', 'tweet-2015']

for j in dataFiles:
	test = pd.read_csv('PreprocessedData/'+j+'-preprocessed.csv', encoding='Latin-1', low_memory=False)

	# print(test.iloc[:, 1])

	all_words = ' '.join([text for text in test.iloc[:, 1]])

	wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

	plt.figure(figsize=(10, 7))
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis('off')
	# plt.show()
	plt.savefig('Plots/'+j+'-wordcloud.png', bbox_inches='tight')
	print(j+' wordcloud saved')
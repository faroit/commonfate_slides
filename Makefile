.PHONY: cleanall

all: images/boxplot.png images/iterations.png

cleanall:
	- rm -f images/boxplot.png images/iterations.png

images/boxplot.png:
	python data/make_boxplot.py data/all.pickle $@

images/iterations.png:
	python data/make_sdriteration.py data/all.pickle $@

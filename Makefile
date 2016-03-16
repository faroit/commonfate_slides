.PHONY: cleanall

all: images/boxplot.svg images/iterations.svg images/gridplot.svg

cleanall:
	- rm -f images/boxplot.svg images/iterations.svg

images/gridplot.svg:
	python data/make_gridplots.py $@

images/boxplot.svg:
	python data/make_boxplot.py data/all.pickle $@

images/iterations.svg:
	python data/make_sdriteration.py data/all.pickle $@

images/patches:
	python data/render_patches.py data/fm_sine.wav

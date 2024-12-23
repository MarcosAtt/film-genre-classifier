# SYNOPSIS:
#
#   make [all]  - makes everything.
#   make TARGET - makes the given target.
#   make clean  - removes all files generated by make.

CXX := g++
CXX_FLAGS := -O3 -Wall -Wextra -std=c++17
CXX_LIB := -shared -fPIC
PYTHONPATH := $(shell python3-config --includes | xargs -n1 | sort -u | xargs |  sed "s# #, #g" | cut -c 3-)
LIB := modules/eigen modules/pybind11/include/ $(PYTHONPATH)
SRC := eigenvalues
LIBRARYFILE := $(SRC)/eigenvalues.cpp
EXT_SUFFIX := $(shell python3-config --extension-suffix)
LIBRARYNAME := eigenvalues$(EXT_SUFFIX)
TARGETDIR := film-classifier
TARGETLIBRARY := $(LIBRARYNAME)
# Library search directories and flags

LDPATHS := $(addprefix -I ,$(LIB) )

SRCS := $(shell find $(SRC) -name '*.cpp')
OBJS := $(subst $(SRC)/,$(BUILD)/,$(addsuffix .o,$(basename $(SRCS))))
DEPS := $(OBJS:.o=.d)

build: clean all

all: $(TARGETLIBRARY)
library: $(TARGETLIBRARY)

$(TARGETLIBRARY):
	@echo "🚧 Building Python Library..."
	$(CXX) $(CXX_FLAGS) $(CXX_LIB) -o $(TARGETDIR)/$(LIBRARYNAME) $(LIBRARYFILE) $(LDPATHS)

# Clean task
.PHONY: clean
clean:
	@echo "🧹 Clearing..."
	rm $(TARGETDIR)/$(LIBRARYNAME) || true

.PHONY: scriptExp
scriptExp:
	jupyter nbconvert --to python notebooks/experimentos.ipynb

.PHONY: bajar_bibliotecas
bajar_bibliotecas:
	git submodule update --init

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... build 🚧"
	@echo "... clean 🧹"
	@echo "... run 🚀 (runs tests)"
	@echo "... bajar_bibliotecas (descarga los submodulos)"

setup-venv:
	./setup-venv.sh

clean-venv:
	./clean-venv.sh

.PHONY: download_data
download_data:
	wget -O data/raw/wiki_movie_plots_deduped_sample.csv https://www.dropbox.com/scl/fi/xnztguety6brdy7t2lfge/wiki_movie_plots_deduped_sample.csv?rlkey=7m867bh7ruilw66qlh7ozl9g4&dl=1

.PHONY : help

MAIN = main
BUILD_DIR = .build
LATEXMK = latexmk

.PHONY: all pdf clean

all: pdf

pdf:
	mkdir -p $(BUILD_DIR)
	TEXINPUTS=.:$(BUILD_DIR): BIBINPUTS=.:$(BUILD_DIR): BSTINPUTS=.:$(BUILD_DIR): \
		$(LATEXMK) -pdf -outdir=$(BUILD_DIR) $(MAIN).tex
	cp $(BUILD_DIR)/$(MAIN).pdf $(MAIN).pdf

clean:
	$(LATEXMK) -C -outdir=$(BUILD_DIR) $(MAIN).tex
	rm -rf $(BUILD_DIR)
	rm -f $(MAIN).pdf

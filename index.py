import generator
import helper

# Generate.
generator.write(inputFilename='drug-brand-names.txt', outputFilename='new-drug-brand-names.txt', scanLength=10, outputLength=10, numberOfOutputs=200, creativity=0.5)

# Clean up generated output.
helper.cleanOutput(outputFilename='new-drug-brand-names.txt', minOutputLength=6)

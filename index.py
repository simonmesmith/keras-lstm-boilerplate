import generator
import helper

# Generate.
generator.write(inputFilename='simon-smith-tweets.txt', outputFilename='new-simon-smith-tweets.txt', scanLength=10, outputLength=500, creativity=0.1)

# Clean up generated output.
helper.cleanOutput(outputFilename='new-simon-smith-tweets.txt', minOutputLength=3)

import generator
import helper

# Generate.
generator.write(inputFilename='movie-names.txt', outputFilename='new-movie-names.txt', scanLength=10, outputLength=500, creativity=0.01, epochs=200)

# Clean up generated output.
helper.cleanOutput(outputFilename='new-movie-names.txt', minOutputLength=3)

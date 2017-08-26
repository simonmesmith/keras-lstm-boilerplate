import generator
import helper

# Generate.
generator.write(input_filename='movie-names.txt', output_filename='new-movie-names.txt', scan_length=10, output_length=500, creativity=0.01, epochs=200)

# Clean up generated output.
helper.clean_output(output_filename='new-movie-names.txt', min_output_length=3)

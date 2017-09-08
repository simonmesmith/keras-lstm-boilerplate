import generator
import helper

# Generate novel strings.
generator.write(
    input_filename='drug-brand-names.txt', # File within the /inputs folder containing the inputs to train on
    output_filename='new-drug-brand-names.txt', # File within the /outputs folder that will hold the outputs
    level='word', # The level at which to create the model and generate strings; valid options are "character" and "word," with "character" being the default
    scan_length=10, # How many strings to scan when predicting the next string
    output_length=500, # How many strings to include in the output
    creativity=0.2, # How creative to be; the higher the number, the more creative the output
    epochs=40 # How many epochs (passes over the input to learn from) to use for training
)

# Clean up generated output.
helper.clean_output(
    output_filename='new-drug-brand-names.txt',  # File within the /outputs folder to clean in
    min_row_length=3 # Minimum length of a row of strings (for example: the shortest length of a generated name)
)

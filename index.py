import generator
import helper

# Generate.
generator.write('company-names.txt', 'new-company-names.txt', 10, 12, 50, .01)

# Clean up generated output.
helper.cleanOutput('new-company-names.txt', 6)

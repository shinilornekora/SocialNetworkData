# Start

DataRefactoring is the main file.
THere we catch the needed fields to process
Then we add one more field 'classes' that is being calculated
Then it goes in file dataWithClasses.

### Files

social_network_users - initial dataset file

data_with_class - file contains needed fields + classes (float field)
cleaned_data - delete absolute values (emissions)
median_replaced_data - classes values replaced by averages (mediums)
data_with_class_int - same but classses field is int now
data_with_sigmoid - same but classes is calculated via sigmoid [maybe drop]
data_with_class_scaled -  normalized values of classes
high_remove_data [drop]
scaled_data - all fields are normalized.

learn.py module is a file that trains model by sklearn
use.py module is a file that works out the predictions
DataPlot module is a module that
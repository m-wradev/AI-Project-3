# AI-Project-3

## How to Compile
Type "make" in the directory containing the makefile to compile the project wit no debug output enabled.

There are four different debug modes that you can set when compiling the project with make:
1. dbg\_bldtree: outputs information related to decision tree creation
2. dbg\_gain: outputs information related to gain calculation for attributes
3. dbg\_ident: outputs information related to the identification of testing data
4. dbg\_all: outputs all previous debug information

## How to Run
id3.exe takes three command-line arguments:
1. Number of features in the feature vectors
2. The text file containing the training data
3. The text file containing the testing data

If I recall correctly, each feature vector for the iris-data dataset contains four attributes, so you would run the program as follows: `./id3 4 training.txt testing.txt`

## Output
id3.exe outputs the number of testing vectors that it correctly identified.

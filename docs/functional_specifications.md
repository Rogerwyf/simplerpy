## Functional Specifications
___
### Background
rpy2 is a well-known bridging package for Python programmers to use R in Python. This package, *simplerpy*, will focus on looking and building upon its interface to make it simpler and easier to use, focusing on functions related to regression and hypothesis testing.

### Users
* Python programmers who would like to run statistical analysis with R in Python
* Researchers who use both languages for statistical research purposes
* Students who use both languages for their coursework

### Data Sources
This package does not require any data. Users may utilize this package on their own datasets for statistical analysis purposes.


### Use Cases
1. Perform quick statistical analysis from R in Python:
   1. Users perform data cleaning and wrangling in Python using Pandas
   2. Instead of saving the results and working with them in R, users can utilize this package to run linear model, hypothesis testing, and analysis of variance in Python
   3. Use `fit(data)` method to fit a model/test
   4. Use `summary()` method to get a quick summary from the fitted model/test similar to one in R
2. Retrieve details from statistical models and tests:
   1. Users are able to get more information from fitted models and tests than they typically do from existing packages (scikit-learn, scipy)
   2. Examples:
      1. Call `LM.p_value()` method for p-values associated with each feature in a linear model
      2. Call `tTest.ci()` method for the confidence interval for a student T-test.
      3. Call `AOV.sum_of_squares()` method for sum of squares in an analysis of variance model
3. Use R objects directly in Python:
   1. Users who are proficient with package rpy2 will have the option to interact directly with R objects
   2. Users can call `r_model_obj()` method to obtain the underlying R object for each component

# Heart Disease Prediction
 
## Overview
This project uses a [heart disease data set from Kaggle](https://www.kaggle.com/fedesoriano/heart-failure-prediction) for classification models in R to help inform prospective business integration with smart health technology. The project includes an R markdown script that runs various classification models. It is intended to determine the best model for identifying heart disease in patients using a dataset with hundreds of observations of patients from around the world.

## Requirements
- R programming language
- R packages: 
  - `car`
  - `dplyr`
  - `tidyr`
  - `caret`
  - `rpart`
  - `rpart.plot`
  - `e1071`
  - `randomForest`
  - `ggplot2`
  - `caTools`
  - `class`
  - `corrplot`
  - `fastDummies`
  - `Hmisc`
  - `reshape`
  - `plotly`
  - `DataExplorer`

## Installation
To run this project, you need to have R installed on your machine. You can download and install R from [CRAN](https://cran.r-project.org/).

### Setting up the R Environment
Once R is installed, open the R console and install the required packages using the following commands:

```R
install.package("ggplot2")
install.package("dplyr")
install.package("caret")
# ... other packages
```

### Initial Run
1. Clone the repository.
```
git clone https://github.gatech.edu/MGT-6203-Fall-2023-Canvas/Team-64.git
```

2. In the repository's Code folder, open the **smart_wellness_heart_disease_analysis.Rmd** file. Set working directory to be the main repository folder either from R Markdown code, or the file options if using RStudio.

3. Run the **smart_wellness_heart_disease_analysis.Rmd** file.

### Directory Structure
```
.
├── Code
│   ├── requirements.txt                                # Required R packages for code
│   └── smart_wellness_heart_disease_analysis.Rmd       # Main code file
├── Data
│   ├── about-heart-data.pdf                            # Data description
│   └── heart.csv                                       # Data set
├── Final Report
├── Other Resources
├── Progress Report
├── Project Proposal
├── Visualizations                                      # Visuals from final report
├── .DS_Store
├── .gitignore
└── README.md
```

### Data Source
fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved August 28, 2023 from https://www.kaggle.com/fedesoriano/heart-failure-prediction.  

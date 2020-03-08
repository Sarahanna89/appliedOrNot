## import both dataframes
job_desc <- read.csv("job_desc.csv")
user <- read.csv("user.csv")

## first overview of data (in particular variable(-names) and scaling)
str(job_desc)
str(user)
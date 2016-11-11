
"""
    NY-SAT-Survey-Scores
    Map SAT Scores to locations along with survey data

    SAT scores by school – SAT scores for each high school in New York City.
    School attendance – attendance information on every school in NYC.
    Math test results – math test results for every school in NYC.
    Class size – class size information for each school in NYC.
    AP test results – Advanced Placement exam results for each high school. Passing AP exams can get you college credit in the US.
    Graduation outcomes – percentage of students who graduated, and other outcome information.
    Demographics – demographic information for each school.
    School survey – surveys of parents, teachers, and students at each school.
    School district maps – contains information on the layout of the school districts, so that we can map them out.
"""

files = ["ap_2010.csv", "class_size.csv", "demographics.csv", "graduation.csv", "hs_directory.csv", "math_test_results.csv", "sat_results.csv"]

"""
x "ap_2010.csv"
x "class_size.csv"
x "demographics.csv"
x "graduation.csv"
x "hs_directory.csv"  which year?
x "math_test_results.csv"
x "sat_results.csv"
"""


# data path:
#
# NY-SAT-Survey-Scores/Survey_Data

import sys
import os
import pandas as pd
import numpy as np
import folium
from folium import plugins

################
# Setup the path to our data files

new_wd = cwd + '\\Survey_Data'
os.chdir(new_wd)
cwd2 = os.getcwd()
#print(cwd2)

###############
# init our various data filenames
#
# hs_directory_2014-2015

files = ["ap_2010.csv",
         "class_size.csv",
         "demographics.csv",
         "graduation.csv",

# Choose 2014-2015:  hs_directory.csv <= hs_directory_2014-2015.csv
         "hs_directory.csv",    # NOTE: choose THIS year 2014-2015
         "math_test_results.csv",
         "sat_results.csv"]

############### 
# Loop through each data file we downloaded.
# Read the file into a Pandas DataFrame.
# Put each DataFrame into a Python dictionary.

data = {}
for f in files:
#    print(f)
#    print("schools/{0}".format(f))
    
    d = pd.read_csv("schools/{0}".format(f))
    data[f.replace(".csv", "")] = d

############### 
# Once we’ve read the data in, we can use the head method 
# on DataFrames to print the first 5 lines of each DataFrame:

#for k,v in data.items():
#    print("\n" + k + "\n")
#    print(v.head(5))


###############
#
#print(data["demographics"]["DBN"].head())
#print("\n\n")

#print("BEFORE\n", data["class_size"].head())


##############
# it looks like the DBN is actually a combination of 
# CSD, BOROUGH, and SCHOOL CODE: for: class_size and hs_directory
#
# So we need to create a new "DBN" column for: class_size and hs_directory

data["class_size"]["DBN"] = data["class_size"].apply(lambda x: "{0:02d}{1}".format(x["CSD"], x["SCHOOL CODE"]), axis=1)
data["hs_directory"]["DBN"] = data["hs_directory"]["dbn"]

#print("AFTER1:\n", data["class_size"].head())

#print("AFTER2:\n", data["hs_directory"].head())

##############
# Load / combine survey data:
#
# Read in the surveys for all schools using the windows-1252 file encoding.
# Read in the surveys for district 75 schools using the windows-1252 file encoding.
# Add a flag that indicates which school district each dataset is for.
# Combine the datasets into one using the concat method on DataFrames.
#
# NOTE: Make sure that the two survey .txt files are named:
# survey_all and: survey_d75
# NOT: survey_all.txt and: survey_d75.txt

#cwd3 = os.getcwd()
#print(cwd3)

survey1 = pd.read_csv("schools/survey_all.txt", delimiter="\t", encoding='windows-1252')
survey2 = pd.read_csv("schools/survey_d75.txt", delimiter="\t", encoding='windows-1252')
survey1["d75"] = False
survey2["d75"] = True
survey = pd.concat([survey1, survey2], axis=0)

# print("SURVEY_CONCAT:\n", survey.head())

#################
# Problem: the survey data has many columns 
# that aren’t very useful to us
#
# We resolve this issue by looking at the data dictionary
# file that we downloaded along with the survey data
#

survey["DBN"] = survey["dbn"]
survey_fields = ["DBN", "rr_s", "rr_t", "rr_p", "N_s", "N_t", "N_p", "saf_p_11", "com_p_11", "eng_p_11", "aca_p_11", "saf_t_11", "com_t_11", "eng_t_10", "aca_t_11", "saf_s_11", "com_s_11", "eng_s_11", "aca_s_11", "saf_tot_11", "com_tot_11", "eng_tot_11", "aca_tot_11",]
survey = survey.loc[:,survey_fields]
data["survey"] = survey


# print("Survey1 Shape:" , survey1.shape)

# print("Survey2 Shape:" , survey2.shape)

# print("Survey Shape:" , survey.shape)


# Making sure you understand what each dataset contains, 
# and what the relevant columns are can save you lots 
# of time and effort later on.

# Now: If we take a look at some of the datasets, 
# including class_size, we’ll immediately see a problem:

data["class_size"].head()

# data["sat_results"].head()

#####################
# we’ll need to find a way to condense datasets 
# like class_size to the point where there’s 
# only a single row per high school.
#
# By restricting each field to a single value, 
# we can filter most of the duplicate rows. 
# In the below code, we:
#
# Only select values from class_size where the GRADE field is 09-12.
# Only select values from class_size where the PROGRAM TYPE field is GEN ED.
# Group the class_size dataset by DBN, and take the average of each column. 
#     (Essentially, we’ll find the average class_size values for each school.)
# Reset the index, so DBN is added back in as a column.

# print("CLASS SIZE BEFORE:\n", data["class_size"].head())

class_size = data["class_size"]
class_size = class_size[class_size["GRADE "] == "09-12"]
class_size = class_size[class_size["PROGRAM TYPE"] == "GEN ED"]
class_size = class_size.groupby("DBN").agg(np.mean)
class_size.reset_index(inplace=True)
data["class_size"] = class_size

# print("CLASS SIZE AFTER:\n", data["class_size"].head())

#####################
# Next, we’ll need to condense the demographics dataset.
#
# For the demographics dataset, there are duplicate rows 
# for each school. We’ll only pick rows where the schoolyear 
# field is the most recent available:
    
demographics = data["demographics"]
demographics = demographics[demographics["schoolyear"] == 20112012]
data["demographics"] = demographics

#print("demographics AFTER:\n", data["demographics"].head())

########################
# Next: We’ll need to condense the math_test_results dataset. 
# This dataset is segmented by Grade and by Year. We can 
# select only a single grade from a single year:

data["math_test_results"] = data["math_test_results"][data["math_test_results"]["Year"] == 2011]
data["math_test_results"] = data["math_test_results"][data["math_test_results"]["Grade"] == '8']

# print("math_test_results AFTER:\n", data["math_test_results"].head())

#######################
# Finally, graduation needs to be condensed:

data["graduation"] = data["graduation"][data["graduation"]["Cohort"] == "2006"]
data["graduation"] = data["graduation"][data["graduation"]["Demographic"] == "Total Cohort"]

# print("graduation AFTER:\n", data["graduation"].head())

####################
# Data cleaning and exploration is critical before 
# working on the meat of the project. Having a good, 
# consistent dataset will help you do your analysis more quickly.

# Computing variables:
# 
# Computing variables can help speed up our analysis 
# by enabling us to make comparisons more quickly, 
# and enable us to make comparisons that we otherwise 
# wouldn’t be able to do

# The first thing we can do is compute a total SAT score 
# from the individual columns SAT Math Avg. Score, 
# SAT Critical Reading Avg. Score, and SAT Writing Avg. Score. 
#
# In the below code, we:
#
# Convert each of the SAT score columns from a string to a number.
# Add together all of the columns to get the sat_score column, which is the total SAT score.

cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']
for c in cols:
#    print("ColName:", c)
# convert_objects has been deprecated:
    data["sat_results"][c] = data["sat_results"][c].convert_objects(convert_numeric=True)
#    str = data["sat_results"][c]
#    print("STR:", str)
#    str = pd.to_numeric(data["sat_results"][c])
#    value = pd.to_numeric(str)
#    print("Val2:", value)
#    data["sat_results"][c] = pd.to_numeric(str)
#    print("ColVal:\n", data["sat_results"][c].head())

data['sat_results']['sat_score'] = data['sat_results'][cols[0]] + data['sat_results'][cols[1]] + data['sat_results'][cols[2]]

#print("sat_results AFTER:\n", data['sat_results']['sat_score'].head())


# NEW method:
#  
# It's just this: data['S1Q2I'] = pd.to_numeric(data['S1Q2I']) 
#

# Warning:
# C:\Anaconda3\lib\site-packages\ipykernel\__main__.py:216: 
# FutureWarning: convert_objects is deprecated.  Use the data-type 
# specific converters pd.to_datetime, pd.to_timedelta and pd.to_numeric.
#
# >>> import pandas as pd
#>>> s = pd.Series(['1.0', '2', -3])
#>>> pd.to_numeric(s)
#>>> s = pd.Series(['apple', '1.0', '2', -3])
#>>> pd.to_numeric(s, errors='ignore')
#>>> pd.to_numeric(s, errors='coerce')


# Next, we’ll need to parse out the coordinate 
# locations of each school, so we can make maps. 
# This will enable us to plot the location of each school. 
#
# In the below code, we:
#
# Parse latitude and longitude columns from the Location 1 column.
# Convert lat and lon to be numeric.

data["hs_directory"]['lat'] = data["hs_directory"]['Location 1'].apply(lambda x: x.split("\n")[-1].replace("(", "").replace(")", "").split(", ")[0])
data["hs_directory"]['lon'] = data["hs_directory"]['Location 1'].apply(lambda x: x.split("\n")[-1].replace("(", "").replace(")", "").split(", ")[1])

for c in ['lat', 'lon']:
    data["hs_directory"][c] = data["hs_directory"][c].convert_objects(convert_numeric=True)


# print("hs_directory_LAT:\n", data["hs_directory"]['lat'].head())
# print("hs_directory_LON:\n", data["hs_directory"]['lon'].head())

##################
# Now, we can print out each dataset to see what we have:
#
#for k,v in data.items():
#    print(k)
#    print(v.head())

####################
# Combining the datasets
#
# Now that we’ve done all the preliminaries
#
# You can read about different types of joins here.

######################
# In the below code, we’ll:
#
#    Loop through each of the items in the data dictionary.
#    Print the number of non-unique DBNs in the item.
#    Decide on a join strategy – inner or outer.
#    Join the item to the DataFrame full using the column DBN.
#

flat_data_names = [k for k,v in data.items()]
flat_data = [data[k] for k in flat_data_names]
full = flat_data[0]
for i, f in enumerate(flat_data[1:]):
    name = flat_data_names[i+1]
    print(name)
    print(len(f["DBN"]) - len(f["DBN"].unique()))
    join_type = "inner"
    if name in ["sat_results", "ap_2010", "graduation"]:
        join_type = "outer"
    if name not in ["math_test_results"]:
        full = full.merge(f, on="DBN", how=join_type)

full.shape

    
####################
# We may want to correlate the Advanced Placement 
# exam results with SAT scores, but we’ll need to 
# first convert those columns to numbers, then fill 
# in any missing values
#

cols = ['AP Test Takers ', 'Total Exams Taken', 'Number of Exams with scores 3 4 or 5']

for col in cols:
    full[col] = full[col].convert_objects(convert_numeric=True)

full[cols] = full[cols].fillna(value=0)


####################
# we’ll need to calculate a school_dist column that 
# indicates the school district of the school. This 
# will enable us to match up school districts and plot
# out district-level statistics using the district 
# maps we downloaded earlier:

full["school_dist"] = full["DBN"].apply(lambda x: x[:2])

#################
# Finally, we’ll need to fill in any missing values 
# in full with the mean of the column, so we can 
# compute correlations:

full = full.fillna(full.mean())

# print(full.head())


##################
# Computing correlations
#
# A good way to explore a dataset and see what 
# columns are related to the one you care about 
# is to compute correlations.
#
# This will tell you which columns are closely
# related to the column you’re interested in. 
# We can do this via the corr method on Pandas DataFrames
#
# The closer to 0 the correlation, the weaker the connection. 
# The closer to 1, the stronger the positive correlation, 
# and the closer to -1, the stronger the negative correlation`:
#
# 

# DATA DOESN'T MATCH
full.corr()['sat_score']



#######################
# In the below code, we:
# Setup a map centered on New York City.
# Add a marker to the map for each high school in the city.
# Display the map.

schools_map = folium.Map(location=[full['lat'].mean(), full['lon'].mean()], zoom_start=10)
marker_cluster = folium.MarkerCluster().add_to(schools_map)
for name, row in full.iterrows():
    folium.Marker([row["lat"], row["lon"]], popup="{0}: {1}".format(row["DBN"], row["school_name"])).add_to(marker_cluster)
schools_map.create_map('schools.html')
schools_map

########################
# This map is helpful, but it’s hard to see w
# where the most schools are in NYC. Instead, we’ll make a heatmap:



schools_heatmap = folium.Map(location=[full['lat'].mean(), full['lon'].mean()], zoom_start=10)
schools_heatmap.add_children(plugins.HeatMap([[row["lat"], row["lon"]] for name, row in full.iterrows()]))
schools_heatmap.save("heatmap.html")
schools_heatmap

###################
# We can compute SAT score by school district, 
# then plot this out on a map. In the below code, we’ll:
#
#    Group full by school district.
#    Compute the average of each column for each school district.
#    Convert the school_dist field to remove leading 0s, so we can match our geograpghic district data.


district_data = full.groupby("school_dist").agg(np.mean)
district_data.reset_index(inplace=True)
district_data["school_dist"] = district_data["school_dist"].apply(lambda x: str(int(x)))


###########################
# We’ll now we able to plot the average SAT score 
# in each school district. In order to do this, 
# we’ll read in data in GeoJSON format to get the 
# shapes of each district, then match each district 
# shape with the SAT score using the school_dist column, 
# then finally create the plot:

def show_district_map(col):
    geo_path = 'schools/districts.geojson'
    districts = folium.Map(location=[full['lat'].mean(), full['lon'].mean()], zoom_start=10)
    districts.geo_json(
        geo_path=geo_path,
        data=district_data,
        columns=['school_dist', col],
        key_on='feature.properties.school_dist',
        fill_color='YlGn',
        fill_opacity=0.7,
        line_opacity=0.2,
    )
    districts.save("districts.html")
    return districts

show_district_map("sat_score")

####### END

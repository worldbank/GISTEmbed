import os
import glob
import pandas as pd


# Get the directory where this script is located
dir_path = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory of the directory where this script is located
parent_dir = os.path.dirname(dir_path)

# Get the list of all the files in the data folder
files = glob.glob("../BingCoronavirusQuerySet/data/*/*.tsv")

# Read the data from all the files and concatenate them into a single dataframe
df = pd.concat([pd.read_csv(f, sep="\t") for f in files], axis=0)

# Get the top explicit queries

all_queries = df[~df["IsImplicitIntent"]].groupby(["Country", "Query"])["PopularityScore"].sum().sort_values(ascending=False)

# Get the list of countries
countries = df["Country"].unique()
valid_countries = set(all_queries.index.levels[0])

# Get the top 100 queries for each country
top_queries = {country: all_queries.loc[country].head(100) for country in countries if country in valid_countries}

# Aggregate the top queries for all countries
top_queries = pd.concat(top_queries.values(), axis=0).sort_values(ascending=False)

top_queries = top_queries.reset_index().groupby("Query")["PopularityScore"].sum().sort_values(ascending=False).reset_index()

# Save the top queries to a file
top_queries.to_json(os.path.join(parent_dir, "data", "queries", "covid_top_queries.json"), orient="records")

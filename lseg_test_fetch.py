import refinitiv.data as rd

# Open LSEG Workspace session
rd.open_session()

# Define and fetch top headlines
headlines = rd.content.news.headlines.Definition(
    query="military OR protest OR default OR strike",
    count=5
)
result = headlines.get_data()
headlines_df = result.data.df

# Print headlines + timestamps cleanly
print("\n📰 LSEG Headlines:\n")
for i, row in headlines_df.iterrows():
    timestamp = row.name  # index is versionCreated
    print(f"📰 {row['headline']}  ({timestamp})")

# Close session
rd.close_session()

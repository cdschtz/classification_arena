# downloads from the GCloud bucket to local data

# download the wikipedia general dataset, ignore macOS DS_Store files
gsutil -m rsync -x ".*\.DS_Store$" -r $BUCKET_DATA_PATH $WIKI_SYNTHETIC_PATH


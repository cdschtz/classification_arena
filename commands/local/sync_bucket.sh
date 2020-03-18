# syncs the GCloud bucket with the local data

# sync the wikipedia general dataset, ignore macOS DS_Store files
gsutil -m rsync -x ".*\.DS_Store$" -r $WIKI_SYNTHETIC_PATH $BUCKET_DATA_PATH


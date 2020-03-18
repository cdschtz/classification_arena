# syncs the GCloud bucket with the local data

# sync the wikipedia general dataset
gsutil -m rsync -x ".*\.DS_Store$" -r $LOCAL_DATA_PATH $BUCKET_DATA_PATH

